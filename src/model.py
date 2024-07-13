
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# Spectral Normalization Layer
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# Model definitions
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        padding_h = kernel_size[0] // 2
        padding_w = kernel_size[1] // 2
        self.padding = (padding_h, padding_w)
        self.bias = bias

        self.conv_i = nn.Conv2d(in_channels=input_dim * 2 + hidden_dim, out_channels=hidden_dim,
                                kernel_size=self.kernel_size, padding=self.padding, bias=bias)
        self.conv_f = nn.Conv2d(in_channels=input_dim * 2 + hidden_dim, out_channels=hidden_dim,
                                kernel_size=self.kernel_size, padding=self.padding, bias=bias)
        self.conv_c = nn.Conv2d(in_channels=input_dim * 2 + hidden_dim, out_channels=hidden_dim,
                                kernel_size=self.kernel_size, padding=self.padding, bias=bias)
        self.conv_o = nn.Conv2d(in_channels=input_dim * 2 + hidden_dim, out_channels=hidden_dim,
                                kernel_size=self.kernel_size, padding=self.padding, bias=bias)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)
        i = torch.sigmoid(self.conv_i(combined))
        f = torch.sigmoid(self.conv_f(combined))
        c = f * c + i * torch.tanh(self.conv_c(combined))
        o = torch.sigmoid(self.conv_o(combined))
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_i.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_i.weight.device))

    def apply_weights_init(self):
        self.conv.apply(weights_init_normal)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SpectrogramProcessor(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=512):
        super(SpectrogramProcessor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)

        def down_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False)]
            if batch_norm:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.down2 = down_block(64, 256, kernel_size=3, dilation=2, padding=2)  # Adjusted kernel size and padding
        self.down3 = down_block(256, 512, kernel_size=3, dilation=4, padding=4)  # Adjusted kernel size and padding
        self.down4 = down_block(512, 512, kernel_size=3, dilation=8, padding=8)  # Adjusted kernel size and padding
        self.down5 = down_block(512, 512, kernel_size=3, dilation=16, padding=16)  # Adjusted kernel size and padding
        self.down6 = down_block(512, 512, kernel_size=3, dilation=32, padding=32)  # Adjusted kernel size and padding
        self.down7 = down_block(512, 512, kernel_size=3, dilation=64, padding=64)  # Adjusted kernel size and padding
        self.down8 = down_block(512, 512, kernel_size=3, dilation=128, padding=128)  # Adjusted kernel size and padding
        self.down9 = down_block(512, 512, kernel_size=3, dilation=256, batch_norm=False,
                                padding=256)  # Adjusted kernel size and padding

        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.self_attention1 = SelfAttention(512)
        self.self_attention2 = SelfAttention(512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.down7(x)
        x = self.down8(x)
        x = self.down9(x)
        x = self.self_attention1(x)
        x = self.self_attention2(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class UNetGenerator(nn.Module):
    def __init__(self, input_channels=4, output_channels=3, hidden_channels=512, lstm_kernel_size=3):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=0.0):
            layers = [nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.PixelShuffle(2)]
            layers.append(nn.BatchNorm2d(out_channels))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(inplace=True))
            return nn.Sequential(*layers)

        self.spectrogram_processor = SpectrogramProcessor(input_channels=1, hidden_dim=hidden_channels)

        self.down1 = down_block(3, 64, batch_norm=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, batch_norm=False)

        self.convlstm = ConvLSTMCell(512, hidden_channels, lstm_kernel_size)

        self.up1 = up_block(hidden_channels, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, output_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

        self.self_attention1 = SelfAttention(512)
        self.self_attention2 = SelfAttention(512)
        self.self_attention3 = SelfAttention(512)

        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.spatial_attention3 = SpatialAttention()

        # Residual Blocks
        self.residual1 = ResidualBlock(64, 64)
        self.residual2 = ResidualBlock(128, 128)
        self.residual3 = ResidualBlock(256, 256)
        self.residual4 = ResidualBlock(512, 512)
        self.residual5 = ResidualBlock(512, 512)
        self.residual6 = ResidualBlock(512, 512)
        self.residual7 = ResidualBlock(512, 512)

        # Dilated Convolution Blocks
        self.dilated1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dilated2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dilated3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dilated4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dilated5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=16, dilation=16)

    def forward(self, x, hidden_state=None):
        spectrogram = x[:, 0:1, :, :]
        frame = x[:, 1:4, :, :]

        d1 = self.residual1(self.down1(frame))
        d2 = self.residual2(self.down2(d1))
        d3 = self.residual3(self.down3(d2))
        d4 = self.residual4(self.down4(d3))
        d5 = self.residual5(self.down5(d4))
        d6 = self.residual6(self.down6(d5))
        d7 = self.residual7(self.down7(d6))
        d8 = self.down8(d7)

        spectrogram_features = self.spectrogram_processor(spectrogram)

        combined_features = torch.cat([d8, spectrogram_features], dim=1)

        if hidden_state is None:
            h, c = self.convlstm.init_hidden(x.size(0), (combined_features.size(2), combined_features.size(3)))
            h, c = self.convlstm(combined_features, h, c)  # Pass hidden_state as separate arguments
        else:
            h, c = hidden_state  # Use the provided hidden state
            h, c = self.convlstm(combined_features, h, c)

        u1 = self.up1(h)

        # Apply spatial attention
        u1 = u1 * self.spatial_attention1(u1)
        u1 = u1 * self.spatial_attention2(u1)
        u1 = u1 * self.spatial_attention3(u1)

        u2 = self.up2(torch.cat([u1, self.self_attention1(d7)], 1))
        u3 = self.up3(torch.cat([u2, self.self_attention2(d6)], 1))
        u4 = self.up4(torch.cat([u3, self.self_attention3(d5)], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8, (h, c)


# Properly initialize the weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
        nn.init.constant_(m.bias.data, 0)
    if classname.find('InstanceNorm') != -1 and hasattr(m, 'weight') and isinstance(m.weight, torch.Tensor):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias.data, 0)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=7):
        super(PatchGANDiscriminator, self).__init__()

        def disc_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.self_attention1 = SelfAttention(512)
        self.self_attention2 = SelfAttention(512)
        
        self.model = nn.Sequential(
            disc_block(input_channels, 64, batch_norm=False),  # First layer should match input_channels
            disc_block(64, 128),
            disc_block(128, 256),
            disc_block(256, 512),
            SelfAttention(512),
            SelfAttention(512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels=7):
        super(MultiScaleDiscriminator, self).__init__()
        self.disc1 = PatchGANDiscriminator(input_channels)
        self.disc2 = PatchGANDiscriminator(input_channels)

    def forward(self, x):
        # print("Size of input to discriminator:", x.size())  # Add this line for debugging
        # print("Datatype of input to discriminator:", x.dtype)  # Add this line for debugging
        x1 = self.disc1(x)
        # print("Size of output from disc1:", x1.size())  # Add this line for debugging
        # print("Datatype of output from disc1:", x1.dtype)  # Add this line for debugging
        x2 = self.disc2(F.avg_pool2d(x, 3, stride=2, padding=[1, 1]))
        return [x1, x2]


def compute_gradient_penalty(D, real_samples, fake_samples):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = [torch.ones_like(d_interpolate, device=device) for d_interpolate in d_interpolates]
    gradients = []
    for d_interpolate, f in zip(d_interpolates, fake):
        gradient = torch.autograd.grad(
            outputs=d_interpolate, inputs=interpolates,
            grad_outputs=f, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients.append(gradient.view(gradient.size(0), -1))
    gradient_penalty = sum(((gradient.norm(2, dim=1) - 1) ** 2).mean() for gradient in gradients)
    return gradient_penalty


class FeatureMatchingLoss(nn.Module):
    def __init__(self, D):
        super(FeatureMatchingLoss, self).__init__()
        self.D = D

    def forward(self, real, fake):
        #print("Size of real before D:", real.size())
        #print("Datatype of real before D:", real.dtype)
        #print("Size of fake before D:", fake.size())
        #print("Datatype of fake before D:", fake.dtype)

        real_features = self.D(real)
        fake_features = self.D(fake)

        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += torch.nn.functional.l1_loss(fake_feat, real_feat)
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg[:36])).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.layers(input)
        target_features = self.layers(target)
        return nn.functional.mse_loss(input_features, target_features)






