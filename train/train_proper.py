'''
Same code from train.ipynb
'''


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import pathlib
# from model import BigramLanguageModel
# from train_model import EncodingDecoding, save_encoder

parent_dir = pathlib.Path(__file__).resolve().parent.parent
models_path = str(parent_dir / "models")
train_path = str(parent_dir / "train")
if models_path not in sys.path:
    sys.path.insert(0, models_path)
if train_path not in sys.path:
    sys.path.insert(0, train_path)

from GPTModel import *
from train_model import save_encoder, EncodingDecoding


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

torch.manual_seed(1337)

batch_size = 32
max_iteration = 5000
block_size = 256
learning_rate = 3e-4
eval_interval = 500
n_embed = 384
dropout = 0.2
n_head = 6
n_layer = 6

print("Loading and preparing data...")

file_path = "reddit_chat.jsonl"
conversations = []
with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        messages = data['message']
        formatted = ""
        for msg in messages:
            if msg['role'] == 'user':
                formatted += f"Human: {msg['content']}\n"
            else:
                formatted += f"Assistant: {msg['content']}\n"
        conversations.append(formatted.strip())

all_text = "\n".join(conversations)
data_encoder = EncodingDecoding(all_text)
vocab_size = len(data_encoder.stoi)
print(f"Vocabulary size: {vocab_size}")

save_encoder(data_encoder, "encoder_vocab.pt")

train_text = "\n\n".join(conversations[:45000])
val_text = "\n\n".join(conversations[45000:])

train_data = torch.tensor(data_encoder.encode(train_text), dtype=torch.long)
val_data = torch.tensor(data_encoder.encode(val_text), dtype=torch.long)

print(f"Train data: {len(train_data):,} tokens")
print(f"Val data: {len(val_data):,} tokens")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(eval_iters=200):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"\nInitializing model...")
model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embed=n_embed,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
)
model.to(device)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\nStarting training for {max_iteration} iterations...")
print(f"Block size: {block_size}, Batch size: {batch_size}, LR: {learning_rate}")
print("-" * 60)

for iteration in range(max_iteration):
    if iteration % eval_interval == 0 or iteration == max_iteration - 1:
        losses = estimate_loss()
        print(f"Step {iteration:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, targets=yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete!")

model_path = "gpt_finetuned_reddit.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

print("\n" + "="*60)
print("Quick generation test:")
print("="*60)

model.eval()
context = torch.tensor([data_encoder.encode("Human: Hello!\nAssistant: ")], dtype=torch.long, device=device)

with torch.no_grad():
    generated = model.generate(context, max_new_tokens=100)

result = data_encoder.decode(generated[0].tolist())
print(result)

