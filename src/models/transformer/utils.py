
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from encoder import EncoderBlock
from decoder import DecoderBlock

def build_encoder_blocks(n_layers: int, d_model: int, n_heads: int, d_feed_fwd: int, dropout: float):
    encoder_blocks = []
    for _ in range(n_layers):
        self_attention_block = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        feed_forward_block = FeedForward(d_model = d_model, d_feed_fwd = d_feed_fwd, dropout = dropout)
        encoder_block = EncoderBlock(
            self_attention_block = self_attention_block,
            feed_forward_block = feed_forward_block,
            dropout = dropout
        )
        encoder_blocks.append(encoder_block)
    
    return encoder_blocks

def build_decoder_blocks(n_layers: int, d_model: int, n_heads: int, d_feed_fwd: int, dropout: float):
    decoder_blocks = []
    for _ in range(n_layers):
        self_attention_block = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        cross_attention_block = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        feed_forward_block = FeedForward(d_model = d_model, d_feed_fwd = d_feed_fwd, dropout = dropout)
        decoder_block = DecoderBlock(
            self_attention_block = self_attention_block,
            cross_attention_block = cross_attention_block,
            feed_forward_block = feed_forward_block,
            dropout = dropout
        )
        decoder_blocks.append(decoder_block)
        
    return decoder_blocks
