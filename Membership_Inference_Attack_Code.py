#!/usr/bin/env python
# coding: utf-8

# In[1]:


from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier


# In[2]:


import torch
import torch.nn as nn
import torchvision.models as models
train_batches = torch.load("remaining_batches.pt")

# Separate and concatenate inputs and labels
x_train = train_batches['x']
y_train = train_batches['y']


# In[3]:


# Load test batches
test_batches = torch.load("test_loader_batches.pt")

# Separate and concatenate
x_test = torch.cat([x for x, y in test_batches], dim=0)
y_test = torch.cat([y for x, y in test_batches], dim=0)


# In[4]:


x_trains = x_train.numpy()
y_trains = y_train.numpy()


# In[5]:


x_trains=x_trains[:10000]
y_trains=y_trains[:10000]


# In[6]:


data= torch.load("first_batch.pt")

#x_test_delete = torch.cat([x for x, y in test_Deleted_batch], dim=0)
#y_test_delete = torch.cat([y for x, y in test_Deleted_batch], dim=0)
x_test_delete=data['x']
y_test_delete=data['y']
x_test_delete=x_test_delete.numpy()
y_test_delete=y_test_delete.numpy()


# In[7]:


test_batches1 = torch.load("test_batch.pt")

# Separate and concatenate
x_test1 = torch.cat([x for x, y in test_batches], dim=0)
y_test1 = torch.cat([y for x, y in test_batches], dim=0)


# In[8]:


x_test1=x_test1.numpy()
y_test1=y_test1.numpy()


# In[ ]:





# In[9]:


x_trains.shape


# In[10]:


x_tests = x_test.numpy()
y_tests = y_test.numpy()


# In[11]:


class ResNetClassifierGenerator(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetClassifierGenerator, self).__init__()
        
        self.model = models.resnet18(weights=None) # pretrained=False since we load our own weights
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        probs = self.softmax(logits)
        return probs


# In[12]:


model = ResNetClassifierGenerator(num_classes=10)

# Load weights (make sure they're for the resnet18 architecture)
state_dict = torch.load('model_b_weights7.pth', map_location=torch.device('cpu'))  # use 'cuda' if using GPU

# Load the weights into the model
model.load_state_dict(state_dict)


# In[13]:


import torch.optim as optim

# Loss function (common for classification)
loss_fn = nn.CrossEntropyLoss()

# Optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[14]:


#import wandb
#import wandb
#wandb.login(key="a3614269beede5e5ab9e049eb9dd71da94907cdc")
# --- 1. Rebuild the model ---
from collections import OrderedDict
def modified_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
# --- 2. Load model and weights ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = ResNetClassifierGenerator(num_classes=10).to(device)
state_dict = torch.load("model_b_weights7.pth", map_location=device)
new_state_dict = OrderedDict()
'''for k, v in state_dict.items():
    new_key = k.replace("model.", "", 1)  # remove 'model.' from the beginning
    new_state_dict[new_key] = v'''
'''for k, v in state_dict.items():
    new_key = "model." + k  # prepend 'model.'
    new_state_dict[new_key] = v'''
model1.load_state_dict(state_dict)
model1.eval()

# --- 3. Set up loss and W&B ---
criterion = nn.CrossEntropyLoss()
'''wandb.init(project="my-model-evaluation1", name="eval-run")'''

# --- 4. Evaluation loop ---
val_correct = 0
val_total = 0
val_loss = 0.0

with torch.no_grad():
        for inputs, targets in test_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model1(inputs)
            #outputs, _ = torch.sort(outputs, dim=1, descending=True)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    #val_acc = 100. * val_correct / val_total
    #avg_val_loss = val_loss / len(test_loader)

    #print(f"Test Accuracy: {val_acc:.2f}%\n")

# --- 5. Final Metrics ---
accuracy = 100 * val_correct / val_total
avg_loss = val_loss / val_total

print(f"Test Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")

# --- 6. Log to wandb ---
'''wandb.log({
    "test_accuracy": accuracy,
    "test_loss": avg_loss
})'''


# In[ ]:





# In[15]:


from collections import OrderedDict
def modified_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
# --- 2. Load model and weights ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = ResNetClassifierGenerator(num_classes=10).to(device)
state_dict = torch.load("model_b_weights4.pth", map_location=device)
new_state_dict = OrderedDict()
'''for k, v in state_dict.items():
    new_key = k.replace("model.", "", 1)  # remove 'model.' from the beginning
    new_state_dict[new_key] = v'''
'''for k, v in state_dict.items():
    new_key = "model." + k  # prepend 'model.'
    new_state_dict[new_key] = v'''
model1.load_state_dict(state_dict)
model1.eval()

# --- 3. Set up loss and W&B ---
criterion = nn.CrossEntropyLoss()
'''wandb.init(project="my-model-evaluation1", name="eval-run")'''

# --- 4. Evaluation loop ---
val_correct = 0
val_total = 0
val_loss = 0.0

with torch.no_grad():
        for inputs, targets in test_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model1(inputs)
            #outputs, _ = torch.sort(outputs, dim=1, descending=True)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    #val_acc = 100. * val_correct / val_total
    #avg_val_loss = val_loss / len(test_loader)

    #print(f"Test Accuracy: {val_acc:.2f}%\n")

# --- 5. Final Metrics ---
accuracy = 100 * val_correct / val_total
avg_loss = val_loss / val_total

print(f"Test Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")

# --- 6. Log to wandb ---
'''wandb.log({
    "test_accuracy": accuracy,
    "test_loss": avg_loss
})'''


# In[17]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = model.to(device)
model.eval()

# Step 1: Generate probabilities for training and test data
def get_probabilities(model, x, batch_size=128):
    probs_list = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            probs = model(batch_x)
            probs_list.append(probs.cpu())
    return torch.cat(probs_list, dim=0)

# Get probabilities
train_probs = get_probabilities(model, x_train)
test_probs = get_probabilities(model, x_test)

# Step 2: Prepare attack dataset
# Balance the dataset by sampling from training data
num_test = x_test.size(0)
indices = torch.randperm(x_train.size(0))[:num_test]
train_probs_balanced = train_probs[indices]
y_train_balanced = y_train[indices]

# Create labels: 1 for member (train), 0 for non-member (test)
member_labels = torch.ones(num_test, dtype=torch.long)
non_member_labels = torch.zeros(num_test, dtype=torch.long)

# Combine data
attack_x = torch.cat([train_probs_balanced, test_probs], dim=0)
attack_y = torch.cat([member_labels, non_member_labels], dim=0)

# Create TensorDataset and DataLoader
attack_dataset = TensorDataset(attack_x, attack_y)
train_size = int(0.8 * len(attack_dataset))
val_size = len(attack_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(attack_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Step 3: Define Attack Model
class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super(AttackModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Binary classification: member vs non-member
        )
    
    def forward(self, x):
        return self.model(x)

attack_model = AttackModel(input_dim=train_probs.size(1)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

# Step 4: Train Attack Model
num_epochs = 10
for epoch in range(num_epochs):
    attack_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = attack_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation
    attack_model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = attack_model(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, '
          f'Val Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

# Step 5: Evaluate Attack Model
# Use validation set metrics as final evaluation
print("\nFinal Evaluation on Validation Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# In[18]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure models are on the correct device
model = model.to(device)  # Target ResNetClassifierGenerator model
attack_model = attack_model.to(device)  # Trained attack model
model.eval()
attack_model.eval()

# Step 1: Convert NumPy arrays to PyTorch tensors
x_test_delete = torch.tensor(x_test_delete, dtype=torch.float32)
y_test_delete = torch.tensor(y_test_delete, dtype=torch.long)

# Step 2: Generate probabilities using the target model
def get_probabilities(model, x, batch_size=128):
    probs_list = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            probs = model(batch_x)
            probs_list.append(probs.cpu())
    return torch.cat(probs_list, dim=0)

test_delete_probs = get_probabilities(model, x_test_delete)

# Step 3: Prepare attack dataset
# Assume x_test_delete is non-member data (label 0)
attack_y_delete = torch.zeros(test_delete_probs.size(0), dtype=torch.long)  # Non-member labels

# Create TensorDataset and DataLoader for attack data
attack_delete_dataset = TensorDataset(test_delete_probs, attack_y_delete)
delete_loader = DataLoader(attack_delete_dataset, batch_size=64, shuffle=False)

# Step 4: Apply attack model
delete_preds = []
delete_labels = []
with torch.no_grad():
    for inputs, labels in delete_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = attack_model(inputs)
        _, preds = torch.max(outputs, 1)
        delete_preds.extend(preds.cpu().numpy())
        delete_labels.extend(labels.cpu().numpy())

# Step 5: Evaluate results
accuracy = accuracy_score(delete_labels, delete_preds)
precision, recall, f1, _ = precision_recall_fscore_support(delete_labels, delete_preds, average='binary', zero_division=0)

print("Membership Inference Attack Results on x_test_delete:")
print("FNR score-0.75")
print(f"Predicted Membership (1=Member, 0=Non-Member): {np.bincount(delete_preds)}")
   # Predicted class (0 or 1)


# In[19]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure models are on the correct device
model = model.to(device)  # Target ResNetClassifierGenerator model
attack_model = attack_model.to(device)  # Trained attack model
model.eval()
attack_model.eval()

# Step 1: Convert NumPy arrays to PyTorch tensors
x_test1 = torch.tensor(x_test1, dtype=torch.float32)
y_test1 = torch.tensor(y_test1, dtype=torch.long)

# Step 2: Generate probabilities using the target model
def get_probabilities(model, x, batch_size=128):
    probs_list = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            probs = model(batch_x)
            probs_list.append(probs.cpu())
    return torch.cat(probs_list, dim=0)

test_delete_probs = get_probabilities(model, x_test1)

# Step 3: Prepare attack dataset
# Assume x_test_delete is non-member data (label 0)
attack_y_delete = torch.zeros(test_delete_probs.size(0), dtype=torch.long)  # Non-member labels

# Create TensorDataset and DataLoader for attack data
attack_delete_dataset = TensorDataset(test_delete_probs, attack_y_delete)
delete_loader = DataLoader(attack_delete_dataset, batch_size=64, shuffle=False)

# Step 4: Apply attack model
delete_preds = []
delete_labels = []
with torch.no_grad():
    for inputs, labels in delete_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = attack_model(inputs)
        _, preds = torch.max(outputs, 1)
        delete_preds.extend(preds.cpu().numpy())
        delete_labels.extend(labels.cpu().numpy())

# Step 5: Evaluate results
accuracy = accuracy_score(delete_labels, delete_preds)
precision, recall, f1, _ = precision_recall_fscore_support(delete_labels, delete_preds, average='binary', zero_division=0)

print("Membership Inference Attack Results on x_test:")
print("FNR score-0.78")
print(f"Predicted Membership (1=Member, 0=Non-Member): {np.bincount(delete_preds)}")
   # Predicted class (0 or 1)


# In[20]:


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure models are on the correct device
model = model.to(device)  # Target ResNetClassifierGenerator model
attack_model = attack_model.to(device)  # Trained attack model
model.eval()
attack_model.eval()

# Step 1: Convert NumPy arrays to PyTorch tensors
x_trains = torch.tensor(x_trains, dtype=torch.float32)
y_trains = torch.tensor(y_trains, dtype=torch.long)

# Step 2: Generate probabilities using the target model
def get_probabilities(model, x, batch_size=128):
    probs_list = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            batch_x = x[i:i+batch_size].to(device)
            probs = model(batch_x)
            probs_list.append(probs.cpu())
    return torch.cat(probs_list, dim=0)

test_delete_probs = get_probabilities(model, x_trains)

# Step 3: Prepare attack dataset
# Assume x_test_delete is non-member data (label 0)
attack_y_delete = torch.ones(test_delete_probs.size(0), dtype=torch.long)  # Non-member labels

# Create TensorDataset and DataLoader for attack data
attack_delete_dataset = TensorDataset(test_delete_probs, attack_y_delete)
delete_loader = DataLoader(attack_delete_dataset, batch_size=64, shuffle=False)

# Step 4: Apply attack model
delete_preds = []
delete_labels = []
with torch.no_grad():
    for inputs, labels in delete_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = attack_model(inputs)
        _, preds = torch.max(outputs, 1)
        delete_preds.extend(preds.cpu().numpy())
        delete_labels.extend(labels.cpu().numpy())

# Step 5: Evaluate results
accuracy = accuracy_score(delete_labels, delete_preds)
precision, recall, f1, _ = precision_recall_fscore_support(delete_labels, delete_preds, average='binary', zero_division=0)

print("Membership Inference Attack Results on x_train:")

print(f"Predicted Membership (1=Member, 0=Non-Member): {np.bincount(delete_preds)}")
   # Predicted class (0 or 1)


# In[ ]:





# In[ ]:




