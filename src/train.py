#!/usr/bin/env python3
import os
import sys
from glob import glob

# Ensure src/ is on PYTHONPATH when running train.py directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from torch.utils.data import random_split
import evaluate
from transformers import (
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
    AutoImageProcessor
)

from dataset import RoadSegDataset

def main():
    # 1) Paths to your data
    IMAGE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'frames')
    MASK_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'masks')

    # 2) Collect files
    imgs  = sorted(glob(os.path.join(IMAGE_DIR, '*.jpg')))
    masks = sorted(glob(os.path.join(MASK_DIR,  '*.jpg')))

    # 3) Match up images and masks by basename
    img_bases  = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    mask_bases = {os.path.splitext(os.path.basename(p))[0] for p in masks}
    common = sorted(img_bases & mask_bases)
    imgs  = [os.path.join(IMAGE_DIR, f + '.jpg') for f in common]
    masks = [os.path.join(MASK_DIR,  f + '.jpg') for f in common]
    print(f"🔹 Using {len(imgs)} image–mask pairs for training")

    # 4) Create dataset and split
    full_ds = RoadSegDataset(imgs, masks,
        model_name='nvidia/segformer-b0-finetuned-cityscapes-512-1024'
    )
    val_size = int(0.1 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])
    print(f"🔹 Train size: {len(train_ds)}, Validation size: {len(val_ds)}")

    # 5) Load the student model (2 classes: background & road)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b0-finetuned-cityscapes-512-1024',
        num_labels=2,
        id2label={0: 'background', 1: 'road'},
        label2id={'background': 0, 'road': 1},
        ignore_mismatched_sizes=True
    ).to(device)

    # 6) Prepare metric
    metric = evaluate.load('mean_iou')
    def compute_metrics(pred):
        preds = pred.predictions.argmax(axis=1)
        return {'mean_iou': metric.compute(predictions=preds, references=pred.label_ids, num_labels=2)['mean_iou']}

    # 7) Training arguments
    args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), '..', 'models', 'my-segformer-road_new'),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=5,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
    )

    # 8) Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=AutoImageProcessor.from_pretrained('nvidia/segformer-b0-finetuned-cityscapes-512-1024'),
        compute_metrics=compute_metrics,
    )

    # 9) Train & save
    trainer.train()
    trainer.save_model()
    print("✅ Training complete. Model saved to", args.output_dir)

if __name__ == '__main__':
    main()
