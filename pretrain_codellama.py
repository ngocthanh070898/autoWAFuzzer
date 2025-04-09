import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LlamaForCausalLM,
    CodeLlamaTokenizer,
    AdamW,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
import os
import argparse

# ========================== ARGUMENT PARSER ==========================
parser = argparse.ArgumentParser(description="Pre-train CodeLlama on attack payloads")
parser.add_argument('--data_path', required=True, help="Path to dataset (TXT/CSV)")
parser.add_argument('--model_name', required=True, help="Path to model directory or HF model name")
parser.add_argument('--output_dir', required=True, help="Directory to save model")
parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate")
parser.add_argument('--max_length', type=int, default=256, help="Max token length")
parser.add_argument('--use_8bit', action='store_true', help="Use 8-bit quantization to reduce memory usage")
args = parser.parse_args()

# ========================== LOAD TOKENIZER & MODEL ==========================
print(f"Loading tokenizer and model from: {args.model_name}")

if not os.path.exists(args.model_name):
    raise FileNotFoundError(f"Model path '{args.model_name}' does not exist!")

tokenizer = CodeLlamaTokenizer.from_pretrained(args.model_name)

# Thêm padding token nếu thiếu
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.use_8bit:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlamaForCausalLM.from_pretrained(args.model_name, quantization_config=quantization_config)
else:
    model = LlamaForCausalLM.from_pretrained(args.model_name)

model.resize_token_embeddings(len(tokenizer))  # Cập nhật model với token mới

# ========================== LOAD DATASET ==========================
print(f"Loading dataset from: {args.data_path}")

def load_payloads(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found!")

    if data_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(data_path)
        return df['payload'].dropna().tolist()
    elif data_path.endswith(".txt"):
        with open(data_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        raise ValueError("Unsupported file format. Use CSV or TXT.")

payloads = load_payloads(args.data_path)

# Custom dataset class
class PayloadDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
      encoding = self.tokenizer(
          self.texts[idx], 
          truncation=True, 
          padding="max_length",  # Đảm bảo tất cả có cùng max_length
          max_length=self.max_length, 
          return_tensors="pt"
      )
      input_ids = encoding["input_ids"].squeeze(0)
      attention_mask = encoding["attention_mask"].squeeze(0)
      return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

dataset = PayloadDataset(payloads, tokenizer, args.max_length)

# Data collator để tự động xử lý padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  
    pad_to_multiple_of=8  # Đảm bảo padding thành bội số của 8
)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)

# ========================== TRAINING LOOP ==========================
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

print("Starting pre-training...")
model.train()

for epoch in range(args.epochs):
    print(f"Epoch {epoch + 1}/{args.epochs}")
    total_loss = 0

    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

# ========================== SAVE MODEL ==========================
print(f"Saving trained model to: {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print("Pre-training completed successfully!")
