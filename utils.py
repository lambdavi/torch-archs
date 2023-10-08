import torch

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            _, pred = model(x).max(dim=1)
            num_correct += (pred==y).sum()
            num_samples += pred.size(0)

        print(f'Accuracy: {num_correct/num_samples*100}')
    
    model.train()
