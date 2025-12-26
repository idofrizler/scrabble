import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. Configuration
DATA_DIR = 'data/tile_ocr'  # Folder with your A, B, C... subfolders
MODEL_SAVE_PATH = 'training/scrabble_net.pth'
IMG_SIZE = 64         # Resize all tiles to 64x64
BATCH_SIZE = 32

# Check for Apple Silicon GPU (MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Data Transforms (Augmentation)
# We add random rotation and slight color jitter to make the model robust
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Scrabble is high contrast, grayscale is fine
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),               # Handle slightly crooked tiles
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Data
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Define a Simple CNN Model
class ScrabbleNet(nn.Module):
    def __init__(self, num_classes):
        super(ScrabbleNet, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize Model
num_classes = len(train_dataset.classes)
model = ScrabbleNet(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Training Loop
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
print(f"Classes: {train_dataset.classes}")
