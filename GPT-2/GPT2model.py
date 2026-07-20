
import torch
import math
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  def __init__(self, n_embd, head_size, block_size):
    super().__init__()
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.head_size = head_size
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self,x):
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)
    B,T,_ = x.shape # Unpack all three dimensions, ignoring the last one if not used.
    scores = Q@K.transpose(-2,-1)
    mask = self.tril[:T, :T] # Use the pre-registered buffer, sliced to current sequence length
    scores= scores.masked_fill(mask==0, float('-inf'))
    scores = scores/math.sqrt(self.head_size)
    scores = F.softmax(scores, dim=-1)
    output = scores@V
    return output

class MultiHeadAttn(nn.Module):
  def __init__(self, n_head, n_embd, block_size):
    super().__init__()
    head_size = n_embd//n_head
    self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_embd, n_embd)
  def forward(self,x):
    output = []
    for head in self.heads:
      output.append(head(x))
    x = torch.cat(output, dim=-1)
    x = self.proj(x)
    return x

class MLP(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.n_embd = n_embd
    self.LinearLayer = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.GELU(),
        nn.Linear(4*n_embd, n_embd)
    )
  def forward(self,x):
    return self.LinearLayer(x)

class Block(nn.Module):
  def __init__(self, n_head, n_embd, block_size):
    super().__init__()

    self.n_embd = n_embd
    self.n_head = n_head
    self.head_size= self.n_embd//self.n_head
    self.dropout = 0.2


    self.multiheadattn = MultiHeadAttn(self.n_head, self.n_embd, block_size)
    self.ln1 = nn.LayerNorm(self.n_embd)
    self.ln2 = nn.LayerNorm(self.n_embd)
    self.mlp = MLP(self.n_embd)
    self.dropout = nn.Dropout(self.dropout)

  def forward(self,x):
    x = x+self.dropout(self.multiheadattn(self.ln1(x)))
    x = x+self.mlp(self.ln2(x))
    return x

class GPT2(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.n_embd = 128
    self.n_head = 4
    self.n_layer = 12
    self.block_size = 128

    self.token_embedding = nn.Embedding(vocab_size, self.n_embd)
    self.position_embedding = nn.Embedding(self.block_size, self.n_embd)
    self.n_layer = 12
    self.ln_f = nn.LayerNorm(self.n_embd)
    self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)
    self.blocks = nn.Sequential(*[Block(self.n_head, self.n_embd, self.block_size) for _ in range(self.n_layer)])


  def forward(self, x, y=None):
    B,T = x.shape
    pos = torch.arange(T, device=x.device)
    x = self.token_embedding(x) + self.position_embedding(pos)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if y is not None:
      B, T, C = logits.shape
      loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      logits, _ = self.forward(idx)
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=-1) # (B, C)
      next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
      idx = torch.cat((idx, next_token), dim=1) # (B, T+1)
    return idx
