import time
import torch
import torch.nn as nn
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import shutil
import cv2
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from tqdm import tqdm
import tensorflow as tf

global device, print_freq
device = torch.device('cuda')


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def __len__(self):
        return self.inp.shape[0]

    def __getitem__(self, index):
        return self.inp[index], self.out[index]


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_step(train_loader, model, criterion, optimizer, epoch, net_info):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if epoch % 10 == 0:
            if i % print_freq == 0:
                curr_lr = optimizer.param_groups[0]['lr']
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'LR: {4}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.7f} ({loss.avg:.6f})\t'.format(
                    epoch, net_info['epochs'], i, len(train_loader), curr_lr,
                    batch_time=batch_time, data_time=data_time, loss=losses))

    return losses.avg


def test_step(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if epoch % 10 == 0:
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.7f} ({loss.avg:.4f})\t'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses))

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_RNN(model, net_info, train_loader, test_loader):
    epochs = net_info['epochs']
    train_losses = np.empty([epochs, 1])
    test_losses = np.empty([epochs, 1])
    min_loss = 1e4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=net_info['lr0'])
    for epoch in tqdm(range(epochs)):
        if epoch >= 50:
            if epoch % 50 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= net_info['lr_decay']
        train_losses_temp = train_step(train_loader, model, criterion, optimizer, epoch, net_info)
        train_losses[epoch] = train_losses_temp
        test_losses_temp = test_step(test_loader, model, criterion, epoch)
        test_losses[epoch] = test_losses_temp
        is_best = test_losses[epoch] > min_loss
        min_loss = min(test_losses[epoch], min_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    test_losses.tofile('test_loss' + str(net_info['train_size']) + '2l50h.csv', sep=",", format="%10.5f")
    train_losses.tofile('train_loss' + str(net_info['train_size']) + '2l50h.csv', sep=",", format="%10.5f")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_losses)
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('MSE')
    ax[0].title.set_text('Training loss')
    ax[0].set_yscale('log')
    ax[1].plot(test_losses)
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('MSE')
    ax[1].title.set_text('Test loss')
    ax[1].set_yscale('log')
    plt.show()

    return model





class cnn_apple(torch.nn.Module):

    def __init__(self):
        super(cnn_apple, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#50*50
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#25*25
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#12*12
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#6*6
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.Linear(2048, 640),
            nn.Linear(640, 35),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.model(x.float())
        return x

def main_program(inp, out, trainer_info, model_name):
    num_sample = trainer_info['train_size']
    inp_train = inp[0:num_sample, :, :, :]
    out_train = out[0:num_sample]
    training_set = MyDataset(inp_train, out_train)

    inp_test = inp[num_sample:inp.shape[0], :, :, :]
    out_test = out[num_sample:inp.shape[0]]
    test_set = MyDataset(inp_test, out_test)

    # create training/test loaders
    train_loader = torch.utils.data.DataLoader(dataset=training_set,batch_size=trainer_info['train_batch'],shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=trainer_info['test_batch'],shuffle=True)

    # train the model
    model = cnn_apple().to(device)
    print(model_name)
    print(model)

    model = train_RNN(model, trainer_info, train_loader, test_loader)
    torch.save(model.state_dict(), model_name)

    return model

training_data = torch.load('training_data.pt')
training_label = torch.load('training_label.pt')

trainer_info = {'epochs': 300,
                'train_size': 30000,
                'train_batch': 100,
                'test_batch': 200,
                'lr0': 1e-5,
                'lr_decay': 0.2}
note = 'cnn_apple'
print_freq = 1000
model_name = 'model' + '_' + str(trainer_info['train_size']) + 'train_' + str(trainer_info['epochs']) + 'ep_' + note
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('model name: ' + model_name)
print(device)


print_freq = 150
print(trainer_info)
model = main_program(training_data, training_label, trainer_info, model_name)



device_name='CPU'#'GPU'
if device_name=='CPU':
    best_model_params = torch.load('checkpoint.pth.tar')['state_dict']
    model = cnn_apple()
    model.load_state_dict(best_model_params)
    data11=torch.load('test_data.pt')
    y=torch.load('test_label.pt')
    yhat=model(data11)
elif device_name=='GPU':
    best_model_params = torch.load('checkpoint.pth.tar', map_location=device)['state_dict']
    model = cnn_apple().to(device)
    model.load_state_dict(best_model_params)
    data11 = torch.load('test_data.pt')
    y = torch.load('test_label.pt')
    yhat=model(data11.to(device)).cpu()
    pass

yhat = yhat.detach().numpy()
y = y.detach().numpy()
yyyy = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
i = 0
for a in yyyy:
    if a == True:
        i = i + 1
print(i / len(yyyy))


