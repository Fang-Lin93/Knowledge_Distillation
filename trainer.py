import torch
import time
from torch import nn
from abc import ABC
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class ModelTrainer:
    """
    model trainer for SL models
    """

    def __init__(self, model, **kwargs):

        self.writer = SummaryWriter()
        self.device = kwargs.get('device', device)
        self.model = model.to(self.device)
        self.N_epochs = kwargs.get('N_epochs', 200)
        self.lr = kwargs.get('lr', 1e-2)
        self.lr_decay = kwargs.get('lr_decay', 0.5)
        self.lr_sch_per = kwargs.get('lr_sch_per', 10)
        self.tag = kwargs.get('tag', '')
        self.l2_regular = kwargs.get('l2_regular', 5e-4)
        self.tolerance = kwargs.get('tolerance', 10)
        self.early_stop = kwargs.get('early_stop', False)
        self.max_acc = -1
        self.min_loss = float('inf')
        self.non_improve = 0

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_regular)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.l2_regular)
        self.teacher = kwargs.get('teacher', None)
        if self.teacher is not None:
            self.teacher.to(self.device)

    def loss_function(self, inputs, targets):
        """
        Overwrite this by loss functions
        :return: loss, predictions
        """
        return self.model.train_loop(inputs, targets, self.teacher)
        # inputs, targets = inputs.to(self.device), targets.to(self.device)
        # logits = self.model(inputs)
        # return F.cross_entropy(logits, targets), logits.argmax(dim=1)

    def train(self, train_loader, val_loader=None):
        """
        example of training classification with cross entropy loss
        It is very different for train() and eval() on a single data!
        """
        num_batches = len(train_loader)
        lr = self.lr
        self.model.train()

        self.max_acc = -1
        self.min_loss = float('inf')
        self.non_improve = 0

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.l2_regular)
        rec_num = 1
        start_time = time.time()
        for epoch in range(self.N_epochs):
            if epoch % self.lr_sch_per == 0:
                lr *= self.lr_decay
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            rec_num = self.train_loop(train_loader, val_loader, epoch, rec_num, num_batches)

            logger.info(f' Epoch {epoch}: {(time.time() - start_time):.3f}s, '
                        f'Remaining: {(time.time() - start_time) / (epoch + 1) * (self.N_epochs - (epoch + 1)):.3f}s')

            # Check early stopping!
            if self.early_stop and self.non_improve >= self.tolerance:
                logger.info("Early Stopping!")
                break
        self.model.eval()

        return self.max_acc

    def train_loop(self, train_loader, val_loader, epoch, rec_num, num_batches):
        logger.debug(f'Epoch {epoch} =======================')
        for batch_id, (x, y) in enumerate(train_loader):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            loss, pred = self.loss_function(x, y)
            acc = pred.eq(y).sum().true_divide(len(y.view(-1)))
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar(f'{self.tag}_loss', loss, rec_num)
            self.writer.add_scalar(f'{self.tag}_acc', acc, rec_num)
            rec_num += 1
            if batch_id % 100 == 0:
                logger.debug(f'Train ({self.tag}) [{batch_id}/{num_batches}]'
                             f'({100. * batch_id / num_batches:.0f}%)]'
                             f'\tLoss: {loss.detach():.6f}\tAcc: {100 * acc.detach():.2f}%')
        if val_loader:
            self.test(val_loader, epoch)
            self.model.train()
        return rec_num

    def test(self, test_loader, epoch=None):
        correct = 0
        loss = 0.
        total = 0.
        self.model.eval()
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                b_loss, pred = self.loss_function(x, y)
                loss += b_loss
                correct += pred.eq(y).sum()
                total += y.view(-1).size(0)

            avg_loss, acc = loss / len(test_loader), correct / total

            if acc > self.max_acc:
                self.max_acc = acc

            if avg_loss < self.min_loss:
                self.min_loss = avg_loss
                self.non_improve = 0
            else:
                self.non_improve += 1

            logger.info(f'Test Loss: {avg_loss:.2f} '
                        f'Test Accuracy: {100 * acc:.2f}%')
            if epoch is not None:
                self.writer.add_scalar(f'Test_loss {self.tag}', avg_loss, epoch)
                self.writer.add_scalar(f'Test_Acc {self.tag}', acc, epoch)
