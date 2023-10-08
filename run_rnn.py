import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from models.rnn import *
from tqdm import tqdm
from utils import check_accuracy

# TRANSFORMATION
transform = transforms.Compose([
    transforms.ToTensor(),
])

if torch.cuda.is_available():
    choice = "cuda"
elif torch.backends.mps.is_available():
    choice = "mps"
else:
    choice = "cpu"

device = torch.device(choice)

# HYPERPARAMS
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2 

# LOAD DATASET 
trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# INITIALISE MODEL
loss_fn = torch.nn.CrossEntropyLoss()

model = RNN(device, input_size, hidden_size, num_layers, sequence_length, num_classes).to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# TRAIN 

for e in range(num_epochs):
    for idx, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.squeeze(dim=1).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.item())
    check_accuracy(test_loader, model, device)

