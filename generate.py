import torch
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPT

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

model = MiniGPT(vocab_size=tokenizer.get_vocab_size()).to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

prompt = "Once upon a time"
tokens = tokenizer.encode(prompt).ids
input_ids = torch.tensor([tokens], device=DEVICE)

with torch.no_grad():
    for _ in range(100):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

print(tokenizer.decode(input_ids[0].tolist()))
