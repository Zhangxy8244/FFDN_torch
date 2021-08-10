import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from FFDN import FusedFuzzyDeepNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_num = 5
input_dim = 28 * 28
fuzz_dim = 100
num_class = 10
batch_size = 32
learning_rate = 10e-5

mnist_dataset = datasets.MNIST('', train=True,
                               transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]),
                               download=True)
train_set, valid_set = random_split(mnist_dataset, [50000, 10000])

train_set_loader = DataLoader(train_set, batch_size=batch_size)
valid_set_loader = DataLoader(valid_set, batch_size=batch_size)

model = FusedFuzzyDeepNet(input_dim, fuzz_dim, num_class).to(device)

min_valid_loss = np.inf
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    train_loss = 0.0
    for data, labels in train_set_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        target = model(data)
        loss = criterion(target, labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.item() * data.size(0)

    valid_loss = 0.0
    model.eval()

    for data, labels in valid_set_loader:
        data, labels = data.to(device), labels.to(device)
        target = model(data)
        loss = criterion(target, labels)
        valid_loss = loss.item() * data.size(0)

    print('Epoch: {:d} - training loss: {:.6f} - validation loss: {:.6f}'.format(epoch, train_loss, valid_loss))
