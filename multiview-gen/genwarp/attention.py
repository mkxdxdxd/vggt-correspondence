import torch
from diffusers.models.attention import Attention
import torch.nn.functional as F
from torch import nn

x = torch.randn(64,64,4)
torch.save(x, 'tensor.pt')

import pdb; pdb.set_trace()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define key features (20 tokens, each with 256 channels)
keys = torch.randn(20, 256).to(device)  # (20, 256)

# Define query features (20 sets of 30 tokens, each with 256 channels)
queries = torch.randn(20, 30, 256).to(device)  # (20, 30, 256)

# Define the value features (can be the same as keys, but let's assume they differ)
values = torch.randn(20, 256).to(device)  # (20, 256)

# Expand keys and values to match the shape of queries (broadcast)
keys_expanded = keys.unsqueeze(1).repeat(1, 30, 1)  # (20, 30, 256)
values_expanded = values.unsqueeze(1).repeat(1, 30, 1)  # (20, 30, 256)

# Use the diffusers Attention class (simple attention mechanism)
# Create an Attention module
attention_layer = Attention(
    query_dim=256,  # The dimension of the query (number of channels)
    heads=8,        # Number of attention heads
    dim_head=32     # The dimension of each attention head (256 / 8 = 32)
).to(device)

# Flatten batch for attention (diffusers' Attention class expects shape [batch_size, seq_len, channels])
queries_flattened = queries.view(-1, 30, 256)  # (20*30, 256)
keys_flattened = keys_expanded.view(-1, 30, 256)  # (20*30, 256)
values_flattened = values_expanded.view(-1, 30, 256)  # (20*30, 256)

import pdb; pdb.set_trace()

# Apply the attention mechanism
# The Attention class in diffusers expects queries and context, where context serves as the keys and values
attention_output_one = attention_layer(queries_flattened, context=keys_flattened)  # Cross-attention

# Reshape the attention output back to (20, 30, 256)
attention_output_one = attention_output_one.view(20, 30, 256)

# Print the shape of the result
print(attention_output_one.shape)  # Expected output: torch.Size([20, 30, 256])

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Linear layers for projecting queries, keys, and values
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(value_dim, query_dim)

        # Output projection
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query, key, value):
        # Project queries, keys, and values
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.query_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, value)

        # Project the output
        attn_output = self.out_proj(attn_output)

        return attn_output

cross_attention = CrossAttention(query_dim=256, key_dim=256, value_dim=256, num_heads=8).to(device)

attention_output = cross_attention(queries, keys_expanded, values_expanded)

import pdb; pdb.set_trace()