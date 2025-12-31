import torch
import torch.nn.functional as F
import json
from model import BigramLanguageModel
from train_model import EncodingDecoding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load encoder
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

# Load model
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
print("Model loaded!\n")

def generate(model, context, max_tokens=150, temperature=0.7, rep_penalty=1.2):
    generated = context.clone()
    for i in range(max_tokens):
        idx_cond = generated[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        recent_tokens = generated[0, -20:].tolist() if generated.shape[1] > 20 else generated[0].tolist()
        for token in set(recent_tokens):
            logits[0, token] /= rep_penalty
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
        if i > 10:
            text_so_far = data_encoder.decode(generated[0].tolist())
            if "\nHuman:" in text_so_far:
                break
    return generated

# Test different question types
test_questions = [
    # Casual/Reddit-style (should work best)
    ("What kind of phone(s) do you guys have?", "Casual Tech"),
    ("How's your day going?", "Casual Greeting"),
    ("What do you think about this?", "Opinion"),
    ("Anyone else experience this?", "Community Question"),
    
    # Technology
    ("What's the best phone right now?", "Tech Recommendation"),
    ("How do I fix this issue?", "Tech Help"),
    
    # Personal/Conversational
    ("What are you up to?", "Personal"),
    ("How are you doing?", "Greeting"),
    
    # Questions (might be weaker)
    ("What is machine learning?", "Technical Definition"),
    ("Explain quantum physics", "Complex Topic"),
]

print("="*70)
print("Testing different question types to see what works best:")
print("="*70)

for question, category in test_questions:
    prompt = f"Human: {question}\nAssistant:"
    context = torch.tensor([data_encoder.encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = generate(model, context, max_tokens=100, temperature=0.7, rep_penalty=1.2)
    
    full_text = data_encoder.decode(output[0].tolist())
    
    if "Assistant:" in full_text:
        parts = full_text.split("Assistant:", 1)
        if len(parts) > 1:
            assistant_response = parts[1]
            if "\nHuman:" in assistant_response:
                assistant_response = assistant_response.split("\nHuman:")[0]
            assistant_response = assistant_response.strip()
            
            print(f"\n[{category}]")
            print(f"Q: {question}")
            print(f"A: {assistant_response[:150]}")
            print("-" * 70)

