import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from modified_blip2 import ModifiedBLIP2  # Import modified BLIP-2 model
import argparse

class RVGDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, masks_path, captions_path, max_samples, transform=None):
        self.images = torch.load(images_path)[0:max_samples]
        self.masks = torch.load(masks_path)[0:max_samples]
        self.captions = torch.load(captions_path)[0:max_samples]  # Loaded as strings
        self.transform = transform

        # Preprocess images and masks to float32 tensors
        self.images = [image.to(dtype=torch.float32) for image in self.images]
        self.masks = [mask.to(dtype=torch.float32).repeat(3, 1, 1) for mask in self.masks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        caption = self.captions[idx]

        return image, mask, caption

def train(model, dataloader, optimizer, device, max_seq_len=32):
    """Training loop."""
    model.train()
    total_loss = 0

    for images, masks, captions in tqdm(dataloader):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        text_input = ["" for caption in captions]
        outputs = model(image=images, mask=masks, text_input = text_input, text_output = captions, max_seq_len=max_seq_len)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, device, max_seq_len=32):
    """Validation loop."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks, captions in tqdm(dataloader):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            text_input = ["" for caption in captions]
            outputs = model(image=images, mask=masks, text_input = text_input, text_output = captions, max_seq_len=max_seq_len)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main(args):
    # Load the pretrained BLIP-2 model from LAVIS library
    pretrained_blip2, _, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=args.device
    )

    # Initialize the modified BLIP-2 model
    model = ModifiedBLIP2(pretrained_blip2).to(args.device)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # DataLoader for training and validation
    train_dataset = RVGDataset(args.train_images, args.train_masks, args.train_captions, max_samples = 5000)
    val_dataset = RVGDataset(args.val_images, args.val_masks, args.val_captions, max_samples = 100)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, args.device)
        val_loss = validate(model, val_loader, args.device)

        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", type=str, required=True, help="Path to training images .pth file")
    parser.add_argument("--train_masks", type=str, required=True, help="Path to training masks .pth file")
    parser.add_argument("--train_captions", type=str, required=True, help="Path to training captions .pth file")
    parser.add_argument("--val_images", type=str, required=True, help="Path to validation images .pth file")
    parser.add_argument("--val_masks", type=str, required=True, help="Path to validation masks .pth file")
    parser.add_argument("--val_captions", type=str, required=True, help="Path to validation captions .pth file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()

    main(args)
