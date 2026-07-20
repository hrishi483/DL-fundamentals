import torch
from GPT2model import GPT2
from main import data, vocab_size, decode
from config import batch_size, block_size, max_iters, learning_rate, eval_interval, checkpoint_interval

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = GPT2(vocab_size).to(device)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split='train'):
    data_source = train_data if split=='train' else val_data
    starts = torch.randint(0, len(data_source) - block_size, (batch_size,))

    x,y = list(), list()
    for start in starts:
        x.append(data_source[start:start+block_size])
        y.append(data_source[start+1:start+block_size+1])

    #Stack all x
    x = torch.stack(x)
    #Stack all y
    y = torch.stack(y)

    return x.to(device), y.to(device)

#Generate sample text
@torch.no_grad()
def generate_sample(step):
    model.eval()

    # Start with a single token (index 0)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=127)
    generated_text = decode(generated[0].tolist())

    with open("generated_samples.txt", "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Iteration: {step}\n")
        f.write("=" * 80 + "\n\n")
        f.write(generated_text)
        f.write("\n\n")

    model.train()

#Eval function:
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        split_losses = []
        for _ in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#Training loop
model.train()
for step in range(max_iters):

    x, y = get_batch("train")
    print(x.shape, y.shape)
    logits, loss = model(x,y)
    print(step, "->", loss)
    optimizer.zero_grad()

    assert loss is not None
    loss.backward()
    optimizer.step()

    #Compute eval
    if step % eval_interval == 0:
        losses = estimate_loss()

        print("*" * 50)
        print(f"Step {step}")
        print(f"Train Loss: {losses['train']:.4f}")
        print(f"Val Loss:   {losses['val']:.4f}")

        generate_sample(step)

    if step%checkpoint_interval==0:
        torch.save(model.state_dict(), f"checkpoint_{step}.pt")