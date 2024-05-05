import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from input_embeddings import InputEmbeddings
from positional_encoder import PositionalEncoder
from projection_layer import ProjectionLayer
from utils import build_encoder_blocks, build_decoder_blocks

class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        state_embed: InputEmbeddings,
        action_embed: InputEmbeddings,
        timestep_embed: PositionalEncoder,
        projection_layer: ProjectionLayer
    ):
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        self.timestep_embed = timestep_embed

        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, state, timestep, src_mask):
        state = self.state_embed(state)
        state = self.timestep_embed(state, timestep)
        x = self.encoder(state, src_mask)

        return x

    def decode(self, encoder_output, src_mask, action, timestep, target_mask):
        action = self.action_embed(action)
        action = self.timestep_embed(state, timestep)
        
        return self.decoder(x = action, encoder_output = encoder_output, src_mask = src_mask, target_mask = target_mask)

    def project(self, x):
        return self.projection_layer(x)

    @staticmethod
    def build(
        state_size: int,
        action_size: int,
        max_ep_len: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_feed_fwd: int = 2048
    ):
        state_embed = InputEmbeddings(input_size = state_size, d_model = d_model)
        action_embed = InputEmbeddings(input_size = action_size, d_model = d_model)

        timestep_embed = PositionalEncoder(d_model = d_model, max_ep_len = max_ep_len, dropout = dropout)        

        encoder_blocks = build_encoder_blocks(n_layers, d_model, n_heads, d_feed_fwd, dropout)
        decoder_blocks = build_decoder_blocks(n_layers, d_model, n_heads, d_feed_fwd, dropout)
        
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))
        
        projection_layer = ProjectionLayer(d_model = d_model, action_size = action_size)
        
        transformer = Transformer(
            encoder=encoder,
            decoder=decoder,
            state_embed=state_embed,
            action_embed=action_embed,
            timestep_embed=timestep_embed,
            projection_layer=projection_layer
        )
        
        for parameter in transformer.parameters():
            if parameter.dim() > 1: nn.init.xavier_uniform_(parameter)

        return transformer

if __name__ == '__main__':
    transformer = Transformer.build(state_size = 438, action_size = 4, max_ep_len = 20)
    print("transformer built", transformer)
