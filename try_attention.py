import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, heads=1):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, query, key, value):
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(K.size(-1))))
        attention = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, V)
        return self.fc_out(out)

def try_cross_attention():
    fA = torch.rand(3, 256)
    fB = torch.rand(3, 256)

    feature_dim = fA.size(-1)  # Assuming fA and fB have the same dimension
    cross_attention = CrossAttention(feature_dim)

    # Compute feature difference and concatenation
    feature_diff = fA - fB
    features_concat = torch.cat([fA, fB], dim=-1)

    # Optionally, you might want to project the concatenated features back to the original dimension
    # if you wish to keep the dimensionality consistent
    proj_layer = nn.Linear(2 * feature_dim, feature_dim)
    #features_concat_proj = proj_layer(features_concat)
    features_concat_proj = proj_layer(features_concat)

    # Use feature difference as query and concatenated features as key-value pairs
    combined_features = cross_attention(feature_diff, features_concat_proj, features_concat_proj)

    # combined_features now contains the attention-combined representation
    print(combined_features.size())


import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_dim, dropout_rate):
        super(TransformerEncoder, self).__init__()

        # Layer normalization layers
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout_rate)

        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Apply layer normalization
        x_norm = self.norm1(x)

        # Multi-head attention
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)

        # Add residual connection
        x = x + attn_output

        # Apply second layer normalization
        x_norm = self.norm2(x)

        # MLP
        mlp_output = self.mlp(x_norm)

        # Add second residual connection
        out = x + mlp_output

        return out


# Example usage
embed_size = 512  # Size of each embedding vector
num_heads = 8  # Number of attention heads
hidden_dim = 2048  # Dimension of the hidden layer in MLP
dropout_rate = 0.1  # Dropout rate

# Create an encoder block
encoder = TransformerEncoder(embed_size, num_heads, hidden_dim, dropout_rate)

# Example input (batch size = 10, sequence length = 20, embedding size = 512)
example_input = torch.randn(10, 20, embed_size)

# Forward pass
output = encoder(example_input)
print(output.shape)  # Output shape will be the same as input shape

