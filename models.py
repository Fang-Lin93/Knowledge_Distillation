
import torch
from torch import nn
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class NN(nn.Module):
    def __init__(self, input_c, output_c, n_layers=2, hidden_c=16):
        super().__init__()

        layers = [nn.Linear(input_c, hidden_c), nn.ReLU()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden_c, hidden_c), nn.ReLU()]

        layers.append(nn.Linear(hidden_c, output_c))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def train_loop(self, inputs, targets, device):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = self(inputs)
        return F.cross_entropy(logits, targets), logits.argmax(dim=1)

    def test(self, data_loader):
        self.eval()
        with torch.no_grad():
            N, correct = 0, 0
            for x, y in data_loader:
                y_hat = self(x).softmax(dim=-1).argmax(dim=-1)

                correct += y_hat.eq(y).sum()
                N += y.size(0)

            acc = correct / N

            print(f'Acc. = {acc:.4f} ({correct}/{N})')

            return acc


class Twin(nn.Module):
    def __init__(self, input_c, output_c, T=1):
        super().__init__()
        self.sm = NN(input_c, output_c, hidden_c=16)
        self.lm = NN(input_c, output_c, hidden_c=256)
        self.distill = True
        self.T = T

    def forward(self, x):

        if self.distill:
            return self.sm(x.view(x.size(0), -1))
        return self.lm(x.view(x.size(0), -1))

    def train_loop(self, inputs, targets, device):
        inputs, targets = inputs.to(device), targets.to(device)

        logit_s, logit_l = self.sm(inputs), self.lm(inputs)

        loss = F.cross_entropy(logit_s, targets) + F.cross_entropy(logit_l, targets)
        loss += - ((logit_l / self.T).softmax(dim=-1) * (logit_s / self.T).log_softmax(dim=-1)).sum(dim=-1).mean()
        loss += - ((logit_s / self.T).softmax(dim=-1) * (logit_l / self.T).log_softmax(dim=-1)).sum(dim=-1).mean()
        # two-sided
        return loss, logit_s.argmax(dim=1)

    def test(self, data_loader):
        self.eval()
        with torch.no_grad():
            N, correct_s, correct_l = 0, 0, 0
            for x, y in data_loader:
                y_hat_s = self.sm(x).softmax(dim=-1).argmax(dim=-1)
                y_hat_l = self.lm(x).softmax(dim=-1).argmax(dim=-1)

                correct_s += y_hat_s.eq(y).sum()
                correct_l += y_hat_l.eq(y).sum()
                N += y.size(0)

            acc_s = correct_s / N

            acc_l = correct_l / N

            print(f'Acc.  small={acc_s:.4f} ({correct_s}/{N})\n'
                  f'Acc.  large={acc_l:.4f} ({correct_l}/{N})')

            return acc_s, acc_l


if __name__ == '__main__':
    import torchvision
    from trainer import ModelTrainer

    data_path = '/Users/fanglinjiajie/locals/datasets/'

    TRANS = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    BATCH_SIZE = 200

    train_set = torchvision.datasets.MNIST(data_path, train=True, transform=TRANS, target_transform=None,
                                           download=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = torchvision.datasets.MNIST(data_path, train=False, transform=TRANS, target_transform=None,
                                          download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # s = NN(784, 10, hidden_c=16)
    """
    s = NN(784, 10, hidden_c=16)
    Best = 0.9527999758720398
    
    s = NN(784, 10, hidden_c=256)
    Best = 0.9818000197410583
    """
    s = Twin(784, 10, T=2)

    trainer = ModelTrainer(s, N_epochs=50, early_stop=False)

    trainer.train(train_loader, test_loader)
    print(f'Best = {trainer.max_acc}')







