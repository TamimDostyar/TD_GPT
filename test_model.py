import torch
import torch.nn.functional as F
import json
from model import BigramLanguageModel
from train_model import load_encoder, EncodingDecoding

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    data_encoder = load_encoder("encoder_vocab.pt")
except (KeyError, ValueError, FileNotFoundError):
    with open("reddit_chat.jsonl", 'r') as f:
        conversations = [
            "\n".join(
                f"{'Human' if msg['role']=='user' else 'Assistant'}: {msg['content']}" 
                for msg in json.loads(line)['message']
            ) 
            for line in f
        ]
    all_text = "\n".join(conversations)
    data_encoder = EncodingDecoding(all_text)

vocab_size = len(data_encoder.stoi)

model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embed=384,
    block_size=256,
    n_head=6,
    n_layer=6,
    dropout=0.2,
    device=device
)
model.load_state_dict(torch.load("gpt_finetuned_reddit.pt", map_location=device))
model.to(device)
model.eval()

def generate(model, context, max_tokens=100, temperature=0.5, top_k=50, rep_penalty=1.2):
    generated = context.clone()
    for _ in range(max_tokens):
        idx_cond = generated[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        recent_tokens = generated[0, -20:].tolist() if generated.shape[1] > 20 else generated[0].tolist()
        for token in set(recent_tokens):
            logits[0, token] /= rep_penalty
        logits = logits / temperature
        if top_k and top_k > 0:
            top_logits, top_indices = torch.topk(logits, top_k)
            probs = F.softmax(top_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(1, next_token_idx)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        text_so_far = data_encoder.decode(generated[0].tolist())
        if "\nHuman:" in text_so_far:
            break
    return generated

question = "AI takes the world"
prompt = f"Human: {question}\nAssistant:"
context = torch.tensor([data_encoder.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    output = generate(model, context)

full_text = data_encoder.decode(output[0].tolist())
if "Assistant:" in full_text:
    assistant_response = full_text.split("Assistant:", 1)[1].split("\nHuman:")[0].strip()
    print(f"Q: {question}")
    print(f"A: {assistant_response}")
