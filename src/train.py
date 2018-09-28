# MIT License
# 
# Copyright (c) 2018 Roland Zimmermann
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from emomatch_dataset import EmoMatchDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from vggvox import VGGVox
import net_sphere
from MobileNetV2 import MobileNetV2
from torch import optim
from tensorboardX import SummaryWriter
import copy
from time import gmtime, strftime
import shutil
import io
import os

class VNet(torch.nn.Module):
    def __init__(self, encoder, num_seg, bi_lstm=False, feature=True):
        super(VNet, self).__init__()
        self.num_seg = num_seg
        self.bi_lstm = bi_lstm
        self.encoder = encoder
        self.linear = torch.nn.Linear(512,2)
        self.tanh = torch.nn.Tanh()
        self.avgPool = torch.nn.AvgPool2d((num_seg,1), stride=1)
        self.dropout = torch.nn.Dropout(0.3)  
        self.feature = feature
        self.LSTM = torch.nn.LSTM(1280, 512, 1, batch_first = True, dropout=0.2, bidirectional=self.bi_lstm)  # Input dim, hidden dim, num_layer
        for name, param in self.LSTM.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
        
    def sequentialLSTM(self, input, hidden=None):
        input_lstm = input.view([-1, self.num_seg, input.shape[1]])
        batch_size = input_lstm.shape[0]
        feature_size = input_lstm.shape[2]

        self.LSTM.flatten_parameters()
            
        output_lstm, hidden = self.LSTM(input_lstm)
        if self.bi_lstm:
             output_lstm = output_lstm.contiguous().view(batch_size, output_lstm.size(1), 2, -1).sum(2).view(batch_size, output_lstm.size(1), -1) 

        # avarage the output of LSTM
        output_lstm = output_lstm.view(batch_size, 1, self.num_seg, -1)
        out = self.avgPool(output_lstm)
        out = out.view(batch_size,-1)
        return out
    
    def forward(self, x):
        x = x.view((x.shape[0]*x.shape[1], *x.shape[2:]))

        x = self.encoder(x)
        x = self.dropout(x)
        x = self.sequentialLSTM(x)
        if self.feature == True: return x

        x = self.linear(x)

        return x

class DAVNet(torch.nn.Module):
    def __init__(self, avnet, gpu_ids=[0]):
        super().__init__()
        self.avnet = nn.DataParallel(avnet, device_ids=gpu_ids)

        self.criterion_CE = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.avnet(x)

    def loss(self, prediction, target):
        return self.criterion_CE(prediction, target)
        
    def eval(self):
        self.avnet.eval()

    def train(self):
        self.avnet.train()

    def accuracy(self, prediction, target):
        target = target.contiguous().view(-1).type(torch.LongTensor)
        prediction = torch.argmax(prediction, dim=1).contiguous().view(-1).type(torch.LongTensor)
        return torch.sum(prediction==target).type(torch.FloatTensor) / float(len(target))

class AVNet(torch.nn.Module):
    def __init__(self, anet, vnet):
        super(AVNet, self).__init__()
        self.anet = anet
        self.vnet = vnet
        self.bn = torch.nn.BatchNorm1d(256)
        self.fc1 = torch.nn.Linear(4096, 128)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(256, 2)
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)

        self.criterion_CE = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        xa, xv = x

        xa = self.anet(xa)
        xv = self.vnet(xv)
        
        xa = self.dropout(xa)
        xv = self.dropout(xv)

        f1 = self.relu(self.fc1(xa))
        f2 = self.relu(self.fc2(xv))

        f = torch.cat((f1, f2), 1)
        
        y = self.softmax(self.fc3(f))

        return y

    def loss(self, prediction, target):
        return self.criterion_CE(prediction, target)
        
    def accuracy(self, prediction, target):
        target = target.view(-1).type(torch.LongTensor)
        prediction = torch.argmax(prediction, dim=1).view(-1).type(torch.LongTensor)
        return torch.sum(prediction==target).type(torch.FloatTensor) / float(len(target))

def random_fliplr(x):
    if np.random.rand() < 0.5:
        return fliplr(x)
    else:
        return x

def fliplr(x):
    return x[:, :, :, ::-1]

def random_noise(x, strength=1):
    noise = np.random.normal(0.0, strength, size=x.shape[-2:])
    
    return np.clip(x + noise, 0.0, 255.0).astype(np.int)

class CombineEmoMatchDataset:
    def __init__(self, batch_size, size, n_video_samples, img_mean_std, aud_mean_std, validation=False):
        def image_transform(x):
            if x is not None:
                return (x - 128.0)/128.0
            return None

        def image_augmentate_transform(x):
            if x is not None:
                x = random_fliplr(x)
                x = random_noise(x, 10)

                return (x - 128.0)/128.0
            return None

        def stft_transform(x):
            if x is not None:
                x = (x.transpose()-aud_mean_std[0])/aud_mean_std[1]
                x = x.transpose()

            return x

        def none_aware_collate_fn(batch):
            batch = [x for x in batch if x ]
            return default_collate(batch)

        ds_p = EmoMatchDataset('/workspace/data/VoxCeleb/dev/', n_video_samples=n_video_samples, image_transform=image_transform if validation else image_augmentate_transform, stft_transform=stft_transform, size=size)
        ds_n = copy.deepcopy(ds_p)
        ds_n.map_correct = False 
        print('Dataset\'s length', len(ds_n))
        dl_p = DataLoader(ds_p, batch_size=batch_size//2, shuffle=True, collate_fn=none_aware_collate_fn, num_workers=16, pin_memory=True)#8
        dl_n = DataLoader(ds_n, batch_size=batch_size//2, shuffle=True, collate_fn=none_aware_collate_fn, num_workers=16, pin_memory=True)
    
        self._dataloaders = (dl_p, dl_n)

    def __len__(self):
        return min([len(x) for x in self._dataloaders])

    def get_dataloader(self):
        return zip(*self._dataloaders)

def train(dataloader, avnet, optimizer, writer, validation_dataset, log_frequency, n_global_step):
    losses = []
    accuracies = []
    avnet.train() 
    pbar = tqdm(dataloader.get_dataloader(), total=len(dataloader), unit="batch")

    def training_step(n_step, item_p_n, n_global_step):       
        x_img = (item_p_n[0][0][0].cuda(), item_p_n[1][0][0].cuda())
        x_aud = (item_p_n[0][0][1].cuda(), item_p_n[1][0][1].cuda())
        y = (item_p_n[0][1].cuda(), item_p_n[1][1].cuda())
        del item_p_n

        x_img = torch.cat(x_img)
        x_aud = torch.cat(x_aud)
        y = torch.cat(y)
        x = (x_img, x_aud)

        optimizer.zero_grad()

        y = y.view(-1)
        y_pred = avnet(x)

        loss = avnet.loss(y_pred, y)
        accuracy = avnet.accuracy(y_pred, y)

        loss.backward()
        optimizer.step()

        loss = loss.item()
        accuracy = accuracy.item()

        pbar.set_description('Loss: {0:.6f}, Accuracy: {1:.6f}'.format(loss, accuracy))

        losses.append(loss)
        accuracies.append(accuracy)

        if len(losses) == log_frequency:
            writer.add_scalar('training/loss', np.mean(losses), n_global_step)
            writer.add_scalar('training/accuracy', np.mean(accuracies), n_global_step)

            losses.clear()
            accuracies.clear()

    for n_step, item_p_n in enumerate(pbar):
        n_global_step += 1
        training_step(n_step, item_p_n, n_global_step)

        if (n_step+1) % 1001 == 0:
            loss, accuracy = validate(validation_dataset, avnet, writer, n_global_step)
            print("Validation: Loss={0:.4f}, Accuracy={1:.4f}".format(loss, accuracy))
            avnet.train() 

    pbar.close()
    del pbar

    return n_global_step

def validate(dataloader, avnet, writer, n_global_step):
    losses = []
    accuracies = []
    avnet.eval()
    pbar = tqdm(dataloader.get_dataloader(), total=len(dataloader), unit="batch")

    def validation_step(n_step, item_p_n):
        x_img = (item_p_n[0][0][0].cuda(), item_p_n[1][0][0].cuda())
        x_aud = (item_p_n[0][0][1].cuda(), item_p_n[1][0][1].cuda())
        y = (item_p_n[0][1].cuda(), item_p_n[1][1].cuda())
        del item_p_n

        x_img = torch.cat(x_img)
        x_aud = torch.cat(x_aud)
        y = torch.cat(y)
        x = (x_img, x_aud)

        y = y.view(-1)
        y_pred = avnet(x)

        loss = avnet.loss(y_pred, y)
        accuracy = avnet.accuracy(y_pred, y)
        loss = loss.item()
        accuracy = accuracy.item()

        pbar.set_description('Loss: {0:.6f}, Accuracy: {1:.6f}'.format(loss, accuracy))

        losses.append(loss)
        accuracies.append(accuracy)

    for n_step, item_p_n in enumerate(pbar):       
        validation_step(n_step, item_p_n)

    pbar.close()
    del pbar

    writer.add_scalar('validation/loss', np.mean(losses), n_global_step)
    writer.add_scalar('validation/accuracy', np.mean(accuracies), n_global_step)

    return np.mean(losses), np.mean(accuracies)

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    batch_size = 12 # 16 # 2
    n_video_samples = 48 # 128
    log_frequency = 100
    lrs = [5e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
    gpu_ids = [0, 1, 2, 3]

    # get time for saving the model later on with an unique name 
    start_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    img_mean_std = np.load('img_norms.npy')
    aud_mean_std = np.load('aud_norms.npy')

    img_mean_std = [x for x in img_mean_std]
    aud_mean_std = [x for x in aud_mean_std]

    anet = VGGVox(n=11, num_classes=None, aux_logits=True)

    encoder=MobileNetV2(n_class=None)
    encoder.load_state_dict(torch.load('mobilenet_v2.pth.tar'), strict=False)
    encoder.cuda()
    vnet = VNet(encoder=encoder, num_seg=n_video_samples)
 
    avnet = AVNet(anet, vnet)
    avnet.cuda(gpu_ids[0])
    avnet = DAVNet(avnet, gpu_ids)

    writer = SummaryWriter()

    n_global_step = 0

    training_dataset = CombineEmoMatchDataset(batch_size=batch_size, size=10000, n_video_samples=n_video_samples, img_mean_std=img_mean_std, aud_mean_std=aud_mean_std)
    validation_dataset = CombineEmoMatchDataset(batch_size=batch_size, size=-500, n_video_samples=n_video_samples, img_mean_std=img_mean_std, aud_mean_std=aud_mean_std, validation=True)
    loss, accuracy = validate(validation_dataset, avnet, writer, n_global_step)
    print("Validation: Loss={0:.4f}, Accuracy={1:.4f}".format(loss, accuracy))
    
    optimizer = optim.SGD(avnet.parameters(), lr=lrs[0])
    for epoch in tqdm(range(10), position=0, unit="epoch"):
        # optimizer = optim.SGD(avnet.parameters(), lr=lrs[epoch])

        n_global_step = train(training_dataset, avnet, optimizer, writer, validation_dataset, log_frequency, n_global_step)

        save_checkpoint(
            {
            'epoch': epoch + 1,
            'global_step': n_global_step,
            'state_dict': avnet.state_dict(),
            'optimizer' : optimizer.state_dict(),
            },
            is_best=False,
            filename='checkpoints/{0}_Epoch_{1}_Step_{2}.pth'.format(start_time, epoch, n_global_step))

    writer.close()
if __name__ == '__main__':
    main()
