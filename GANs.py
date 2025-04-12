#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
'''import wandb
wandb.login(key="your key")'''


# In[2]:


import torch
import torch.nn as nn
import torchvision.models as models

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
        


# In[3]:


class ProbabilityDiscriminator(nn.Module):
    def __init__(self, input_dim=10):
        super(ProbabilityDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(32, 1),
           # nn.Sigmoid()  # outputs probability: real (1) or fake (0)
        )

    def forward(self, prob_vec):
        return self.model(prob_vec)


# In[4]:


def compute_gradient_penalty(D, real_probs, fake_probs):
    batch_size = real_probs.size(0)
    alpha = torch.rand(batch_size, 1).to(real_probs.device)

    interpolates = (alpha * real_probs + (1 - alpha) * fake_probs).requires_grad_(True)

    d_interpolates = D(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp


# In[5]:


from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_a = ResNetClassifierGenerator(num_classes=10)

state_dict = torch.load("resnet18_weights.pth", map_location=device)

# Example: Add 'model.' prefix to all keys (if your model expects that)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = "model." + k  # or adjust to match exactly
    new_state_dict[new_key] = v

model_a.load_state_dict(new_state_dict)
model_a.to(device)
model_a.eval()

# Prepare model_b for training
model_b = ResNetClassifierGenerator(num_classes=10)
pretrained_weights = torch.load('resnet18_weights.pth', map_location='cpu')

# Load weights into the model
model_b.model.load_state_dict(pretrained_weights)
model_b.to(device)
model_b.train()


# In[6]:


images,labels=torch.load('test_batch.pt')
with torch.no_grad():
    real_prob=model_a(images)
    real_prob, _ = torch.sort(real_prob, dim=1, descending=True)
torch.save(real_prob, 'real_prob_test.pt')


# In[7]:


from torch.utils.data import TensorDataset, DataLoader
prob_tensor = torch.load('real_prob_test.pt')  
dataset = TensorDataset(prob_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# In[8]:


for batch in dataloader:
    probs = batch[0]  # since batch is a tuple
    print(probs.shape)


# In[9]:


import torch.optim as optim

D=ProbabilityDiscriminator()
D_optimizer = optim.Adam(D.parameters(), lr=0.001)
model_b_optimizer = optim.Adam(model_b.model.parameters(), lr=0.001)


# In[10]:


num_epochs = 10
n_critic = 5
λ = 10
'''wandb.init(project="Gans", name="run_name", config={
    "num_epochs": num_epochs,
    "n_critic": n_critic,
    "lambda": λ,
})'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(num_epochs):
    for i, real_probs in enumerate(dataloader):
        real_probs = real_probs[0].to(device) 
        batch_size = real_probs.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(n_critic):
            data = torch.load('First_batch.pt')
            z=data['x']
            z = z.to(device)
            fake_probs = model_b(z).detach()
            fake_probs, _ = torch.sort(fake_probs, dim=1, descending=True)

            D_real = D(real_probs)
            D_fake = D(fake_probs)

            gp = compute_gradient_penalty(D, real_probs, fake_probs)

            D_loss = D_fake.mean() - D_real.mean() + λ * gp

            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        data = torch.load('First_batch.pt')
        z=data['x']
        z = z.to(device)
        fake_probs = model_b(z)
        fake_probs, _ = torch.sort(fake_probs, dim=1, descending=True)
        model_b_loss = -D(fake_probs).mean()

        model_b_optimizer.zero_grad()
        model_b_loss.backward()
        model_b_optimizer.step()

        # Log metrics to wandb
        '''wandb.log({
            "Epoch": epoch + 1,
            "D_loss": D_loss.item(),
            "model_b_loss": model_b_loss.item(),
        })'''

    print(f"[Epoch {epoch+1}/{num_epochs}] D_loss: {D_loss.item():.4f} | model_b_loss: {model_b_loss.item():.4f}")
torch.save(model_b.state_dict(), "model_b_weights7.pth")


# In[ ]:





# In[ ]:




