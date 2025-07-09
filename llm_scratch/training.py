import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from trainable_model import Transformer, ModelArgs
from safetensors.torch import save_file

class TxtDataset(Dataset):
    def __init__(self, txt_file, tokenizer, max_length):
        with open(txt_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        self.examples = []
        for line in lines:
            tokens = tokenizer.encode(line, truncation=True, max_length=max_length)
            if len(tokens) < 2:
                continue
            self.examples.append(tokens)
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.zeros(self.max_length, dtype=torch.long)
        x[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)

        y = torch.full((self.max_length,), -1, dtype=torch.long)
        y[:len(tokens)-1] = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "gpt2"
    input_file = "/Users/shashi.b/Desktop/Synergy2/LLM/llm_scratch/input.txt"
    max_length = 128
    batch_size = 8
    num_epochs = 50
    lr = 1e-4

    args = ModelArgs()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = Transformer(args).to(device)

    dataset = TxtDataset(input_file, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Dataloader batches: {len(dataloader)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    os.makedirs("checkpoints", exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        for step, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            if logits.dim() != 3:
                raise ValueError(f"Model output shape incorrect: got {logits.shape}, expected [batch, seq_len, vocab]")

            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)

            loss = criterion(logits_flat, y_flat)
            print(f"Step {step}: Loss {loss.item()} | Backward call.")
            loss.backward()
            optimizer.step()

            if step % 100 == 0 or step == len(dataloader) - 1:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        checkpoint_path = f"checkpoints/model_epoch_{epoch+1:02d}.safetensors"
        save_file(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    print("Training completed successfully.")


if __name__ == "__main__":
    train()
