import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- STEP 1: DEFINE DATASET FOR FOLDER-BASED IMAGES/MASKS ---
class FolderLaneDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Custom Dataset for reading images and lane masks from folders.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # List all image files in the directory
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((256, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, (512, 256))

        # --- AUTO-GENERATE REAL MASK FROM WHITE LANE MARKINGS ---
        gray = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (512, 256)), cv2.COLOR_BGR2GRAY)
        # Threshold to isolate bright white/yellow lane lines
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        mask = thresh # Use actual detected bright pixels as the mask
        # -------------------------------------------------------

        # Normalize to tensors
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# --- STEP 2: CORRECTED LIGHTWEIGHT UNET ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2, 2) # 256x512 -> 128x256
        
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2, 2) # 128x256 -> 64x128

        # Decoder
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2) # 64x128 -> 128x256
        # Input channels set to 48 (16 from up1 + 32 from skip connection e2)
        self.dec1 = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(16, 1, 2, stride=2)  # 128x256 -> 256x512

    def forward(self, x):
        e1 = self.enc1(x)         # Size: 256 x 512, Channels: 16
        p1 = self.pool1(e1)       # Size: 128 x 256, Channels: 16
        
        e2 = self.enc2(p1)        # Size: 128 x 256, Channels: 32
        p2 = self.pool2(e2)       # Size: 64 x 128, Channels: 32

        u1 = self.up1(p2)         # Size: 128 x 256, Channels: 16
        u1 = torch.cat([u1, e2], dim=1) # Channels: 16 + 32 = 48
        d1 = self.dec1(u1)
        
        out = torch.sigmoid(self.up2(d1)) # Size: 256 x 512, Channels: 1
        return out

# --- STEP 3: TRAINING LOOP ---
def main():
    # Point this to your actual folder containing images shown in the explorer
    IMAGE_DIR = "./dataset/archive (1)/road_line_images/road_line_images"  # update to your exact subfolder path

    print("Initializing dataset from folder...")
    dataset = FolderLaneDataset(image_dir=IMAGE_DIR)
    
    if len(dataset) == 0:
        print(f"Error: No images found in {IMAGE_DIR}. Check your folder path.")
        return

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training loop...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "tusimple_lane_model.pth")
    print("Training complete! Model saved as 'tusimple_lane_model.pth'.")

if __name__ == "__main__":
    main()