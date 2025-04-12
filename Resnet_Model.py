#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import wandb
wandb.login(key="Your Key")


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
])


# In[4]:


'''import py7zr
import os

# Path to your .7z file
archive_path = 'train.7z'

# Directory where you want to extract the files
extract_dir = os.path.join(os.getcwd(), 'Extracted_Images')

# Create the directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Open the archive
with py7zr.SevenZipFile(archive_path, mode='r') as archive:
    # Extract all files to the specified directory
    archive.extractall(path=extract_dir)

# List the extracted files
extracted_files = os.listdir(extract_dir)
print("Extracted files:", extracted_files)'''


# In[5]:


class SimpleLabelDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(labels_file)  # CSV with 'id' and 'label' columns
        self.img_dir = img_dir
        self.transform = transform

        # Encode string labels to integers
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.labels_df['label'].unique()))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        # Store image paths and numeric labels
        self.image_ids = self.labels_df['id'].tolist()
        self.encoded_labels = [self.label_to_idx[label] for label in self.labels_df['label']]

    def __len__(self):
        return len(self.encoded_labels)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.encoded_labels[idx]
        return image, label


# In[6]:


dataset = SimpleLabelDataset(labels_file="trainLabels.csv", img_dir="Extracted_Images/train", transform=transform_train)


# In[7]:


image, label = dataset[3]
print(f"Label (str): {label}")
print(f"Label (str): {dataset.idx_to_label[label]}")
print(f"Image shape: {image.shape}")


# In[8]:


from torch.utils.data import random_split

# Assume dataset is already created
 # replace with your actual dataset

# Set the split ratio
val_ratio = 0.2  # 20% validation, 80% training
total_size = len(dataset)
val_size = int(val_ratio * total_size)
train_size = total_size - val_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# In[9]:


from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# In[21]:


def save_dataloader_batches(dataloader, filename):
    batches = [batch for batch in dataloader]
    torch.save(batches, filename)

# Example usage:
save_dataloader_batches(train_loader, "train_loader_batches.pt")
save_dataloader_batches(test_loader, "test_loader_batches.pt")


# In[22]:


first_batch = None
remaining_batches = []

# Go through the loader
for i, (x_batch, y_batch) in enumerate(train_loader):
    if i == 0:
        first_batch = (x_batch, y_batch)
    else:
        remaining_batches.append((x_batch, y_batch))

# Save first batch to file
x_first, y_first = first_batch
torch.save({'x': x_first, 'y': y_first}, 'first_batch.pt')

# Combine all remaining batches into one tensor (optional)
x_remain = torch.cat([x for x, _ in remaining_batches], dim=0)
y_remain = torch.cat([y for _, y in remaining_batches], dim=0)
torch.save({'x': x_remain, 'y': y_remain}, 'remaining_batches.pt')

print("✅ First batch and remaining batches saved as 'first_batch.pt' and 'remaining_batches.pt'")


# In[17]:


import os

# Create folders if not exist
os.makedirs("train_parts", exist_ok=True)
os.makedirs("test_parts", exist_ok=True)

def save_batches(dataloader, prefix, output_dir):
    iterator = iter(dataloader)
    for file_idx in range(3):
        batches = []
        for _ in range(10):
            try:
                batch = next(iterator)
                batches.append(batch)
            except StopIteration:
                break
        torch.save(batches, os.path.join(output_dir, f"{prefix}_part_{file_idx + 1}.pt"))

# Assuming these are defined
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

save_batches(train_loader, "train", "train_parts")
save_batches(test_loader, "test", "test_parts")


# In[10]:


train_iter = iter(train_loader)
train_images, train_labels = next(train_iter)

# Get one batch from test_loader
test_iter = iter(test_loader)
test_images, test_labels = next(test_iter)

# Save the batches
torch.save((train_images, train_labels), 'train_batch.pt')
torch.save((test_images, test_labels), 'test_batch.pt')

print("Saved one batch each from train and test loaders.")


# In[11]:


# 4. Modified ResNet-18
def modified_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = modified_resnet18().to(device)


# In[12]:


from tqdm import tqdm


# In[13]:


# 5. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[14]:


'''# 6. Training Loop
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test Accuracy: {100.*correct/total:.2f}%\n")
'''


# In[15]:


wandb.init(project="your_project_name", config={
    "epochs": 10,
    "batch_size": train_loader.batch_size,
    "optimizer": "Adam",  # or whatever you use
    "loss": "CrossEntropyLoss",
    "architecture": "ResNet18"
})


# In[16]:


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        loop.set_postfix(loss=loss.item(), acc=acc)

    avg_train_loss = train_loss / len(train_loader)

    # 🧪 Evaluation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(test_loader)

    print(f"Test Accuracy: {val_acc:.2f}%\n")

    # 📝 Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": acc,
        "val_loss": avg_val_loss,
        "val_accuracy": val_acc
    })
torch.save(model.state_dict(), "resnet18_weights.pth")


# In[ ]:




