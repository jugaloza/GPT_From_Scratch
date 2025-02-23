import torch.nn as nn
import torch
import torch.nn.functional


class AttentionHead(nn.Module):
    def __init__(self, block_size, n_embed, head_size):
        super().__init__()

        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))
        #print(head_size)
        #print(n_embed)
        self.qw = nn.Linear(n_embed, int(head_size))
        self.kw = nn.Linear(n_embed, int(head_size))
        self.vw = nn.Linear(n_embed, int(head_size))

    def forward(self, x):

        B, T, C = x.shape

        q = self.qw(x) # B, T, head_size
        k = self.kw(x) # B, T, head_size
        v = self.vw(x) # B, T, head_size

        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        wei = torch.nn.functional.softmax(wei, dim=-1)
        out = wei @ v # B, T, head_size
        
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, n_embed, n_layers):
        super().__init__()

        self.module = nn.ModuleList([AttentionHead(block_size, n_embed, n_embed / n_layers) for _ in range(n_layers)])
        self.projs = nn.Linear(int((n_embed / n_layers) * n_layers) , n_embed)

    def forward(self,x):

        #x = self.module(x) 

        x = torch.cat([h(x) for h in self.module], dim = -1) # B, T, n_h * h_s
        x = self.projs(x) # B, T, C

        return x

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self,x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, block_size, n_embed, n_heads):
        super().__init__()
        self.multiheadattn = MultiHeadAttention(block_size, n_embed, n_heads)
        self.ffwd = FeedForward(n_embed)
        self.layernorm1 = nn.LayerNorm(n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        
        x = self.multiheadattn(x)
        x = self.layernorm1(x)
        x = self.ffwd(x)
        x = self.layernorm2(x)

        return x
    

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, n_layer, n_heads):
        super().__init__()
        self.block_size = block_size
        self.word_embedding = nn.Embedding(vocab_size,n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(block_size, n_embed, n_heads) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, vocab_size)
        #self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))



    def forward(self, x, y=None):
        
        B, T = x.shape
        x = self.word_embedding(x) # B, T, embed
        x_pos = self.position_embedding(torch.arange(T)) # B, T, embed

        x = x + x_pos
        #print(x)
        x  = self.blocks(x) # B, T, n_embed
        logits = self.lm_head(x) #B, T, vocab_size

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T,C)

            y = y.view(-1)

            loss = torch.nn.functional.cross_entropy(logits, y)

        #x = self.ffwd(x) # B, T, vocab_size




        return logits, loss
    
    def generate(self, idx, max_num_tokens):

        for _ in range(max_num_tokens):
            crop_idx = idx[:, -self.block_size :]
            logits = self(crop_idx)
            probs = torch.nn.functional.softmax(logits[:,-1, :], dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_idx], dim=1)

        return idx