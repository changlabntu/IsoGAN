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
