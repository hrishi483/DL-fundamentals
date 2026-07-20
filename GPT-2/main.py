import torch

with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
  text = f.read()
print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

#Encoding
encode = lambda s: [stoi[c] for c in s]
#Decoding
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii"))
print(decode(encode("hii there")))

#Convert entire datasset
data = torch.tensor(encode(text), dtype=torch.long)