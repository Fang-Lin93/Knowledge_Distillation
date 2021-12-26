
import torch
import torchvision


data_path = '/Users/fanglinjiajie/locals/datasets/'

TRANS = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # it will rescale 255 -> 1
BATCH_SIZE = 100

train_set = torchvision.datasets.MNIST(data_path, train=True, transform=TRANS, target_transform=None, download=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.MNIST(data_path, train=False, transform=TRANS, target_transform=None, download=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

