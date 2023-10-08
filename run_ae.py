import torch
from models.autoencoder import AutoEncoder, CNNAutoEncoder
from tqdm import tqdm
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
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
input_size = 28

trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False)


# INITIALISE MODEL
loss_fn = torch.nn.MSELoss()
use_cnn = True
if use_cnn:
    model = CNNAutoEncoder()
else:
    model = AutoEncoder(input_size=input_size)
model = model.to(device)

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

n_epochs = 15

# TRAIN 
outputs = []
for e in range(n_epochs):
    for idx, (images, _) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        if not use_cnn:
            images = images.reshape(-1, input_size*input_size)
        output = model(images)
        loss = loss_fn(output, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    outputs.append((e, images, output))
        

    print(loss.item())

for k in range(0, n_epochs, 4):
    plt.figure(figsize=(9,2))
    plt.gray()
    imgs = outputs[k][1].cpu().detach().numpy()
    recon = outputs[k][2].cpu().detach().numpy()

    for i, item in enumerate(imgs):
        if i>=9: break
        plt.subplot(2, 9, i+1)
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i>=9: break
        plt.subplot(2,9,9+i+1)
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])
    plt.show(block=True)