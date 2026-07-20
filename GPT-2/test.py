from GPT2model import GPT2
from main import vocab_size
from main import encode, decode
import torch

model = GPT2(vocab_size)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

idx_list = encode("O my Lord")

# Convert the list of indices to a tensor and add a batch dimension
idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device).unsqueeze(0)

generated_tokens = model.generate(idx_tensor, 10)

# Decode the generated tokens back to a string
print(decode(generated_tokens[0].tolist()))
