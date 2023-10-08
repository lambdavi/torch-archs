import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from models.alexnet import AlexNet
from models.generic_cnn import Gen_CNN
from tqdm import tqdm
from utils import check_accuracy

# LOAD DATASET & TRANSFORMATION
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

if torch.cuda.is_available():
    choice = "cuda"
elif torch.backends.mps.is_available():
    choice = "mps"
else:
    choice = "cpu"

device = torch.device(choice)

bs = 16

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)


# INITIALISE MODEL
loss_fn = torch.nn.CrossEntropyLoss()

model = Gen_CNN(n_classes=10)

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(loss.item())
    check_accuracy(test_loader, model, device)

