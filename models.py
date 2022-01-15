
import torch
from torch import nn
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

EPS = 1e-5


class NN(nn.Module):
    def __init__(self, input_c, output_c, n_layers=2, hidden_c=16, entropy_reg=False, alpha=0.5, T=1):
        super().__init__()

        layers = [nn.Linear(input_c, hidden_c), nn.ReLU()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden_c, hidden_c), nn.ReLU()]

        layers.append(nn.Linear(hidden_c, output_c))
        self.output_c = output_c
        self.net = nn.Sequential(*layers)
        self.entropy_reg = entropy_reg
        self.alpha = alpha
        self.T = T

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def train_loop(self, inputs, targets, teacher=None):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets) if self.alpha < 1 else 0
        if self.entropy_reg:
            loss += - (logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1).mean()

        if self.alpha > 0:
            s_prob = (logits / self.T).softmax(dim=-1)

            if teacher is not None:  # if no teacher -> use uniform
                teacher.eval()
                with torch.no_grad():
                    t_prob = (teacher(inputs) / self.T).softmax(dim=-1)

                loss = (1 - self.alpha) * loss + self.alpha * (t_prob * (t_prob / (EPS+s_prob)).log()).sum(dim=-1).mean()
            else:
                uniform = torch.ones_like(s_prob) / self.output_c
                loss = (1 - self.alpha) * loss + self.alpha * (uniform * (uniform / (EPS+s_prob)).log()).sum(dim=-1).mean()
                   # ((logits_t/self.T).softmax(dim=-1) * (logits/self.T).log_softmax(dim=-1)).sum(dim=-1).mean()

        return loss, logits.argmax(dim=-1)

    def test(self, data_loader):
        self.eval()
        with torch.no_grad():
            N, correct = 0, 0
            for x, y in data_loader:
                y_hat = self(x).softmax(dim=-1).argmax(dim=-1)

                correct += y_hat.eq(y).sum()
                N += y.size(0)

            acc = correct / N

            logger.info(f'Acc. = {acc:.4f} ({correct}/{N})')

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

    def train_loop(self, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)

        logit_s, logit_l = self.sm(inputs), self.lm(inputs)

        loss = F.cross_entropy(logit_s, targets) + F.cross_entropy(logit_l, targets)
        loss += - self.T**2 * \
            ((logit_l / self.T).softmax(dim=-1) * (logit_s / self.T).log_softmax(dim=-1)).sum(dim=-1).mean()
        loss += - self.T**2 * \
            ((logit_s / self.T).softmax(dim=-1) * (logit_l / self.T).log_softmax(dim=-1)).sum(dim=-1).mean()
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


def double_train(tr_loader, val_loader):
    m1, m2 = NN(784, 10, hidden_c=16, T=3).to(device), NN(784, 10, hidden_c=256, T=3).to(device)
    m1_train, m2_train = ModelTrainer(m1, N_epochs=1, teacher=m2), ModelTrainer(m2, N_epochs=1, teacher=m1)

    max_acc = 0
    for i in range(100):
        logger.info(f'Epoch={i}')
        m1_train.train(tr_loader)
        m2_train.train(tr_loader)

        acc = m1.test(val_loader)

        if acc > max_acc:
            max_acc = acc

    logger.info(f'Double model: max acc. = {max_acc}')


if __name__ == '__main__':
    import sys
    import torchvision
    from trainer import ModelTrainer

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    data_path = '/Users/fanglinjiajie/locals/datasets/'

    TRANS = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    BATCH_SIZE = 200

    train_set = torchvision.datasets.MNIST(data_path, train=True, transform=TRANS, target_transform=None,
                                           download=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = torchvision.datasets.MNIST(data_path, train=False, transform=TRANS, target_transform=None,
                                          download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # double_train(train_loader, test_loader)

    s = NN(784, 10, hidden_c=16)
    # s = Twin(784, 10, T=1)

    trainer = ModelTrainer(s, N_epochs=100, early_stop=False)
    trainer.train(train_loader, test_loader)
    print(f'Best = {trainer.max_acc}')

    self_s = NN(784, 10, hidden_c=16, T=3)
    trainer = ModelTrainer(self_s, N_epochs=100, early_stop=False, teacher=s)
    trainer.train(train_loader, test_loader)
    print(f'Best = {trainer.max_acc}')

    """
    s = NN(784, 10, hidden_c=16)
    Raw Best = 0.9595000147819519
    Entropy Best = 0.9556000232696533
    self-learning: Best = 0.9553 -> 0.9603  (+ 0.5 %, T=3); 0.9574 -> 0.9572 (- 0.02 %, T=1) 0.9549
    double = 0.9581 (T=1),  = 0.9608 (T=3)
    self-learning (T = 3): Best = 0.9566 -> 0.9562  ( - 0.04 %)


    s = NN(784, 10, hidden_c=256)
    Raw Best = 0.9832000136375427
    Entropy Best = 0.9850999712944031
    self-learning: Best = 0.9850 ->  ?  0.9859 (+ 0.09 %) not significant !
    self-learning (T = 3): Best = 0.9841 ->  ?  0.9841 (+ 0.0 %)


    Twin
    (tensor(0.9562), tensor(0.9768))
    """









