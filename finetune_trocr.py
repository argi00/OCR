import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# Load processor and model
model_name = "microsoft/trocr-base-stage1"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Set decoder start token id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True)


# If you have GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------------
# 2. Load Dataset
# -----------------------------
DATA_DIR = "./data/synthetic/images"
LABEL_FILE = "./data/synthetic/labels.txt"

# labels.txt format -> "image_path \t text"
with open(LABEL_FILE, encoding="utf-8") as f:
    lines = [line.strip().split("\t") for line in f.readlines()]

# Use the paths as they are in the labels.txt
images = [l[0] for l in lines]  # <-- do NOT join with DATA_DIR
texts = [l[1] for l in lines]

train_imgs, val_imgs, train_txts, val_txts = train_test_split(
    images, texts, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Create a PyTorch Dataset
# -----------------------------
class OcrDataset(Dataset):
    def __init__(self, image_paths, texts, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        text = self.texts[idx]

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(text, padding="max_length", max_length=128, truncation=True).input_ids
        labels = torch.tensor(labels)

        return {"pixel_values": pixel_values, "labels": labels}

train_dataset = OcrDataset(train_imgs, train_txts, processor)
val_dataset = OcrDataset(val_imgs, val_txts, processor)

# -----------------------------
# 4. Define Training Args
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-finetuned-french-lab",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    eval_strategy="steps",
    num_train_epochs=10,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

# -----------------------------
# 5. Define Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# -----------------------------
# 6. Train the model
# -----------------------------
trainer.train()

# -----------------------------
# 7. Save model + processor
# -----------------------------
model.save_pretrained("./trocr-finetuned-french-lab")
processor.save_pretrained("./trocr-finetuned-french-lab")

print("✅ Fine-tuning completed and model saved at ./trocr-finetuned-french-lab")
