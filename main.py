import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from models.alexnet import AlexNet
from tqdm import tqdm

# LOAD DATASET & TRANSFORMATION
transform = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = "mps"
bs = 16

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)


# INITIALISE MODEL
loss_fn = torch.nn.CrossEntropyLoss()

model = AlexNet(num_classes=10)

model = model.to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

n_epochs = 5

# TRAIN 

for e in range(n_epochs):
    for idx, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    print(loss.item())

