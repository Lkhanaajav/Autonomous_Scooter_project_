# src/train_patches.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from patch_utils import make_patches, PatchDataset
from patch_classifier import PatchClassifier

IMG_DIR       = os.path.join("data", "frames")
MASK_DIR      = os.path.join("data", "masks")
PATCH_DIR     = os.path.join("data", "patches")
MODEL_OUT     = os.path.join("models", "patch-classifier.pt")
BATCH_SIZE    = 32
LR            = 1e-4
NUM_EPOCHS    = 5
VAL_SPLIT     = 0.1
FRAME_STEP    = 100
PATCHES_PER_F = 5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
os.makedirs(PATCH_DIR, exist_ok=True)

print("▶︎ Generating patches …")
make_patches(
    img_dir=IMG_DIR,
    mask_dir=MASK_DIR,
    out_dir=PATCH_DIR,
    size=64,
    frame_step=FRAME_STEP,
    patches_per_frame=PATCHES_PER_F
)

print("▶︎ Building patch dataset …")
full_dataset = PatchDataset(PATCH_DIR)
total = len(full_dataset)
val_count = int(total * VAL_SPLIT)
train_count = total - val_count
train_ds, val_ds = random_split(full_dataset, [train_count, val_count])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"   Total patches: {total} → Train: {train_count}, Val: {val_count}")

model     = PatchClassifier(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    run_loss = 0.0
    corr_train = 0
    tot_train = 0
    t0 = time.time()

    for batch in train_loader:
        imgs = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        corr_train += torch.sum(preds == labels).item()
        tot_train += imgs.size(0)

    train_loss = run_loss / tot_train
    train_acc  = corr_train / tot_train
    t1 = time.time()

    model.eval()
    corr_val = 0
    tot_val = 0
    with torch.no_grad():
        for batch in val_loader:
            imgs   = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            corr_val += torch.sum(preds == labels).item()
            tot_val += imgs.size(0)

    val_acc = corr_val / tot_val
    print(
        f"Epoch {epoch}/{NUM_EPOCHS}  "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
        f"Val Acc: {val_acc:.4f}  Time: {(t1-t0):.1f}s"
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ↪︎ Saved best (Val Acc: {val_acc:.4f})\n")

print("▶︎ Training complete. Best Val Acc:", best_val_acc)
print("▶︎ Model saved to:", MODEL_OUT)
