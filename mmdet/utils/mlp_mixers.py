import einops
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.):
        super(MLPBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.Linear1 = nn.Linear(mlp_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.Linear2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        x = self.dropout(x)
        return x

class Mixer_struc(nn.Module):
    def __init__(self, patches, token_dim, dim,channel_dim,dropout = 0.):
        super(Mixer_struc, self).__init__()
        self.patches = patches
        self.channel_dim = channel_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.MLP_block_token = MLPBlock(patches, token_dim, self.dropout)
        self.MLP_block_chan = MLPBlock(patches, channel_dim, self.dropout)
        self.LayerNorm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.LayerNorm(x)
        out = einops.rearrange(out, 'b n d -> b d n')
        out = self.MLP_block_token(out)
        out = einops.rearrange(out, 'b d n -> b n d')
        out += x
        out2 = self.LayerNorm(out)
        out2 = self.MLP_block_chan(out2)
        out2 += out
        return out2




class MLP_Mixer(nn.Module):
    def __init__(self, image_size, patch_size, token_dim, channel_dim, num_classes, dim, num_blocks):
        super(MLP_Mixer, self).__init__()
        n_patches =(image_size//patch_size) **2
        #self.patch_size_embbeder = nn.Conv2d(kernel_size=n_patches, stride=n_patches, in_channels=3, out_channels= dim)
        self.patch_size_embbeder = nn.Conv1d(in_channels=3,out_channels=dim,kernel_size=n_patches)
		
        self.blocks = nn.ModuleList([
            Mixer_struc(patches=n_patches, token_dim=token_dim,channel_dim=channel_dim,dim=dim) for i in range(num_blocks)
        ])

        self.Layernorm1 = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
    def forward(self,x):
        out = self.patch_size_embbeder(x)
        out = einops.rearrange(out,"n c h w -> n (h w) c")
        for block in self.blocks:
            out = block(out)
        out = self.Layernorm1(out)
        out = out.mean(dim = 1) # out（n_sample,dim）
        result = self.classifier(out)
        return result