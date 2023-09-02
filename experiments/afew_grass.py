import sys
import time

sys.path.insert(0,'..')
from pathlib import Path
import os
import random

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data

import spd.nn as nn_spd
import torch.optim as optim

from mat import MatFolder
from spd.optimizers import MixOptimizer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("grasslogs")

def afew(data_loader):

    #main parameters
    lr=5e-2
    data_path='data/afew/'
    n=400 #dimension of the data
    C=7 #number of classes
    batch_size=30 #batch size
    threshold_reeig=1e-4 #threshold for ReEig layer
    epochs=30000

    if not Path(data_path).is_dir():
        print("Error: dataset not found")
        print("Please download and extract the file at the following url: http://www-connex.lip6.fr/~schwander/datasets/afew.tgz")
        sys.exit(2)

    #setup data and model
    class AfewNet(nn.Module):
        def __init__(self):
            super(__class__,self).__init__()
            dim=400
            dim1=200; dim2=100; dim3=50
            classes=7
            self.re=nn_spd.ReEig()
            self.bimaps1 = nn_spd.BiMaps(1,1,300,400)
            self.bimaps2 = nn_spd.BiMaps(1,1,100,150)
            self.qr1 = nn_spd.Qrs()
            self.qr2 = nn_spd.Qrs()

            self.projmap1 = nn_spd.Projmap()
            self.projmap2 = nn_spd.Projmap()
            self.projmap3 = nn_spd.Projmap()
            self.avg1 = nn.AvgPool2d(stride=2,kernel_size=2)
            self.avg2 = nn.AvgPool2d(stride=2,kernel_size=2)
            self.linear=nn.Linear(20000,classes).double()
            self.orth1 = nn_spd.Orthmap()
            self.orth2 = nn_spd.Orthmap()


        def forward(self,x):
            x_spd = self.bimaps1(x)#对应matlab frmap是很普通的权重更新
            x_spd = self.qr1(x_spd)
            x_spd = self.projmap1(x_spd)
            x_spd = self.avg1(x_spd)
            x_spd = self.orth1(x_spd)
            x_spd = self.bimaps2(x_spd)
            x_spd = self.qr2(x_spd)
            x_spd = self.projmap2(x_spd)
            x_spd = self.avg2(x_spd)
            x_spd = self.orth2(x_spd)
            x_spd = self.projmap3(x_spd)


            x_spd = x_spd.view(x_spd.shape[0],-1)
            y=self.linear(x_spd)
            return y
    model=AfewNet()

    #setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # opti = MixOptimizer(model.parameters(),lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=5e-4, momentum=0.0)

    #initial validation accuracy
    loss_val,acc_val=[],[]
    y_true,y_pred=[],[]
    gen = data_loader._test_generator
    # models

    #training loop
    for epoch in range(epochs):

        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        i=0
        for local_batch, local_labels in data_loader._train_generator:
            time_start = time.time()

            # opti.zero_grad()
            optimizer.zero_grad()

            out = model(local_batch)
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1)==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            # opti.step()
            optimizer.step()

            time_end = time.time()
            time_c= time_end - time_start
            print("time cost %.2f loss:%.2f  acc:%.2f" % (float(time_c), float(loss), float(acc)))
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()
        writer.add_scalar("train_loss", loss_train, epoch)
        writer.add_scalar("train_acc",acc_train,epoch)
        writer.flush()
        # validation
        loss_val,acc_val=[],[]
        y_true,y_pred=[],[]
        gen = data_loader._test_generator
        model.eval()
        for local_batch, local_labels in gen:
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels=out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_val.append(loss)
            acc_val.append(acc)
        acc_val = np.asarray(acc_val).mean()
        loss_val = np.asarray(loss_val).mean()
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))
        with open('afewgrass.txt', 'a') as file:
            file.write(str(acc_val) + "\n")
        writer.add_scalar("val_loss", loss_val, epoch)
        writer.add_scalar("val_acc", acc_val, epoch)
        if epoch % 50 == 0:
            th.save(model.state_dict(),'epoch'+str(epoch)+'.pt')


    print('Final validation accuracy: '+str(100*acc_val)+'%')
    return acc_val

if __name__ == "__main__":

    data_path='data/afew/'
    batch_size=30 #batch size

    class DatasetSPD(data.Dataset):
        def __init__(self, path, names):
            self._path = path
            self._names = names
        def __len__(self):
            return len(self._names)
        def __getitem__(self, item):
            x = np.load(self._path + self._names[item])[None, :, :].real
            x = th.from_numpy(x).double()
            y = int(self._names[item].split('.')[0].split('_')[-1])
            y = th.from_numpy(np.array(y)).long()
            return x,y

    class DataLoaderAFEW:
        def __init__(self):

            self._train_generator=data.DataLoader(MatFolder(r"C:\Users\l\Desktop\备份代码\20230829代码\batchspd\batchspd\experiments\data\afew\train"),batch_size=batch_size,shuffle='True',drop_last=True)
            self._test_generator=data.DataLoader(MatFolder(r"C:\Users\l\Desktop\备份代码\20230829代码\batchspd\batchspd\experiments\data\afew\val"),batch_size=batch_size,shuffle='False',drop_last=True)

    afew(DataLoaderAFEW())
