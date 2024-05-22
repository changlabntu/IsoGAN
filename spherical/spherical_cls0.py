import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score


class TiffDataset(Dataset):
    def __init__(self, folder_a, folder_b, transform=None):
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.transform = transform
        self.files_a = sorted(os.listdir(folder_a))
        self.files_b = sorted(os.listdir(folder_b))

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, idx):
        img_a_path = os.path.join(self.folder_a, self.files_a[idx % len(self.files_a)])
        img_b_path = os.path.join(self.folder_b, self.files_b[idx % len(self.files_b)])

        img_a = Image.open(img_a_path)
        img_b = Image.open(img_b_path)

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
])

# Create datasets
folder_a = 'spherical/aeffphi'
folder_b = 'spherical/beffphi'
dataset = TiffDataset(folder_a, folder_b, transform=transform)

# Create dataloader
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load pre-trained VGG-11 model
vgg11 = models.vgg11(pretrained=True)

# Freeze all layers
for param in vgg11.parameters():
    param.requires_grad = False

# Modify the final layer for binary classification (assuming two classes)
num_ftrs = vgg11.classifier[6].in_features
vgg11.classifier[6] = nn.Linear(num_ftrs, 2)  # Modify the output layer to match the number of classes

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vgg11 = vgg11.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg11.classifier[6].parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    vgg11.train()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    for imgs_a, imgs_b in dataloader:
        # Assume the labels are 0 for folder A images and 1 for folder B images
        labels_a = torch.zeros(imgs_a.size(0), dtype=torch.long)
        labels_b = torch.ones(imgs_b.size(0), dtype=torch.long)

        imgs_a, labels_a = imgs_a.to(device), labels_a.to(device)
        imgs_b, labels_b = imgs_b.to(device), labels_b.to(device)

        if 1:
            # Forward pass for folder A images
            outputs_a = vgg11(imgs_a.repeat(1, 3, 1, 1))
            loss_a = criterion(outputs_a, labels_a)
            # Forward pass for folder B images
            outputs_b = vgg11(imgs_b.repeat(1, 3, 1, 1))
            loss_b = criterion(outputs_b, labels_b)
            # Combine losses
            loss = loss_a + loss_b

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Collect labels and outputs for AUC calculation
        all_labels.extend(labels_a.cpu().numpy())
        all_labels.extend(labels_b.cpu().numpy())
        all_outputs.extend(outputs_a.detach().cpu().numpy()[:, 1])
        all_outputs.extend(outputs_b.detach().cpu().numpy()[:, 1])

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_outputs)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}, AUC: {auc:.4f}")

print("Training complete.")
