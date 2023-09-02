import sys
import time

import torch

from spd.functional import StiefelParameter

sys.path.insert(0, '../..')
from pathlib import Path
import os
import random

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data

import spd.nn as nn_spd
import cplx.nn as nn_cplx
from spd.optimizers import MixOptimizer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("attenlogs")

def afew(data_loader):

    #main parameters
    lr=5e-2
    data_path='data/afew/'
    n=400 #dimension of the data
    C=7 #number of classes
    batch_size=30 #batch size
    threshold_reeig=1e-4 #threshold for ReEig layer
    epochs=3000

    if not Path(data_path).is_dir():
        print("Error: dataset not found")
        print("Please download and extract the file at the following url: http://www-connex.lip6.fr/~schwander/datasets/afew.tgz")
        sys.exit(2)

    class SPDTransform(nn.Module):

        def __init__(self, input_size, output_size):
            super(SPDTransform, self).__init__()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.increase_dim = None
            if output_size > input_size:
                self.increase_dim = SPDIncreaseDim(input_size, output_size)
                input_size = output_size
            self.weight = StiefelParameter(torch.DoubleTensor(1,1,input_size, output_size).to(self.device),
                                           requires_grad=True)
            # self.weight = StiefelParameter(torch.DoubleTensor(input_size, output_size).to(self.device),
            #                                requires_grad=True)
            nn.init.orthogonal_(self.weight)
            #一会注释掉，attention的前向传播维度要对应
            #以前遇到过，一旦成为参数向量 就不能更改维度了，
            # self.weight=self.weight.unsqueeze(0).unsqueeze(0)

        def forward(self, input):
            output = input
            if self.increase_dim:
                output = self.increase_dim(output)
            #权重维度可能有问题 后面改 现在走到这里是1，1，200，100
            #正确的这里是200 ，100 所以去掉两个维度
            weight = self.weight.squeeze(0).squeeze(0).unsqueeze(0)
            # weight = self.weight.unsqueeze(0)
            weight = weight.expand(input.size(0), -1, -1)
            output = torch.bmm(weight.transpose(1, 2), torch.bmm(output, weight))

            return output

    class SPDIncreaseDim(nn.Module):

        def __init__(self, input_size, output_size):
            super(SPDIncreaseDim, self).__init__()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.register_buffer('eye', torch.eye(output_size, input_size).to(self.device))
            add = torch.as_tensor([0] * input_size + [1] * (output_size - input_size), dtype=torch.float32)
            add = add.to(self.device)
            self.register_buffer('add', torch.diag(add))

        def forward(self, input):
            eye = self.eye.unsqueeze(0)
            eye = eye.expand(input.size(0), -1, -1)
            add = self.add.unsqueeze(0)
            add = add.expand(input.size(0), -1, -1)

            output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1, 2)))

            return output

    class AttentionManifold(nn.Module):
        def __init__(self, in_embed_size, out_embed_size):
            super(AttentionManifold, self).__init__()

            self.d_in = in_embed_size
            self.d_out = out_embed_size

            self.q_trans = SPDTransform(self.d_in, self.d_out).cpu()
            self.k_trans = SPDTransform(self.d_in, self.d_out).cpu()
            self.v_trans = SPDTransform(self.d_in, self.d_out).cpu()

        def tensor_log(self, t):  # 4dim
            u, s, v = th.svd(t)
            return u @ th.diag_embed(th.log(s)) @ v.permute(0, 1, 3, 2)

        def tensor_exp(self, t):  # 4dim
            # condition: t is symmetric!
            s, u = th.linalg.eigh(t)
            return u @ th.diag_embed(th.exp(s)) @ u.permute(0, 1, 3, 2)

        def log_euclidean_distance(self, A, B):
            inner_term = self.tensor_log(A) - self.tensor_log(B)
            inner_multi = inner_term @ inner_term.permute(0, 1, 3, 2)
            _, s, _ = th.svd(inner_multi)
            final = th.sum(s, dim=-1)
            return final

        def LogEuclideanMean(self, weight, cov):
            # cov:[bs, #p, s, s]
            # weight:[bs, #p, #p]
            bs = cov.shape[0]
            num_p = cov.shape[1]
            size = cov.shape[2]
            cov = self.tensor_log(cov).view(bs, num_p, -1)
            output = weight @ cov  # [bs, #p, -1]
            output = output.view(bs, num_p, size, size)
            return self.tensor_exp(output)

        def forward(self, x, shape=None):
            if len(x.shape) == 3 and shape is not None:
                x = x.view(shape[0], shape[1], self.d_in, self.d_in)
            # x = x.to(th.float)  # patch:[b, #patch, c, c]
            q_list = []
            k_list = []
            v_list = []
            # calculate Q K V
            bs = x.shape[0]
            m = x.shape[1]
            x = x.reshape(bs * m, self.d_in, self.d_in)
            Q = self.q_trans(x).view(bs, m, self.d_out, self.d_out)
            K = self.k_trans(x).view(bs, m, self.d_out, self.d_out)
            V = self.v_trans(x).view(bs, m, self.d_out, self.d_out)

            # calculate the attention score
            Q_expand = Q.repeat(1, V.shape[1], 1, 1)

            K_expand = K.unsqueeze(2).repeat(1, 1, V.shape[1], 1, 1)
            K_expand = K_expand.view(K_expand.shape[0], K_expand.shape[1] * K_expand.shape[2], K_expand.shape[3],
                                     K_expand.shape[4])

            atten_energy = self.log_euclidean_distance(Q_expand, K_expand).view(V.shape[0], V.shape[1], V.shape[1])
            atten_prob = nn.Softmax(dim=-2)(1 / (1 + th.log(1 + atten_energy))).permute(0, 2, 1)  # now row is c.c.

            # calculate outputs(v_i') of attention module
            output = self.LogEuclideanMean(atten_prob, V)

            output = output.view(V.shape[0], V.shape[1], self.d_out, self.d_out)

            shape = list(output.shape[:2])
            shape.append(-1)

            # output = output.contiguous().view(-1, self.d_out, self.d_out)
            return output, shape
    #setup data and model
    class AfewNet(nn.Module):
        def __init__(self):
            super(__class__,self).__init__()
            dim=400
            dim1=200; dim2=100; dim3=50
            classes=7
            self.re=nn_spd.ReEig()
            # self.batchnorm1=nn_spd.BatchNormSPD(dim1)
            # self.bimap2=nn_spd.BiMap(1,1,dim1,dim2)
            # self.batchnorm2=nn_spd.BatchNormSPD(dim2)
            # self.bimap3=nn_spd.BiMap(1,1,dim2,dim3)
            # self.batchnorm3=nn_spd.BatchNormSPD(dim3)
            self.logeig=nn_spd.LogEig()
            self.linear=nn.Linear(dim3**2,classes).double()

            # self.bimap6 = nn_spd.BiMap(1,1,400,100)
            # self.batchnorm5 = nn_spd.BatchNormSPD(100)
            # self.bimap4 = nn_spd.BiMap(1,1,200,400)
            # self.batchnorm4 =nn_spd.BatchNormSPD(400)
            # self.bimap5 = nn_spd.BiMap(1, 1, 400, 200)
            # self.batchnorm5 = nn_spd.BatchNormSPD(200)
            self.att1 = AttentionManifold(200, 100)
            self.bimap11 = nn_spd.BiMap(1,1,400,200)
            self.bimap12 = nn_spd.BiMap(1,1,400,200)
            self.bimap13 = nn_spd.BiMap(1,1,400,200)
            self.bimap2 = nn_spd.BiMap(1,1,100,50)
            self.bn1 = nn_spd.BatchNormSPD(200)
            self.bn2 = nn_spd.BatchNormSPD(200)
            self.bn3 = nn_spd.BatchNormSPD(200)
        def forward(self,x):
            #30 1 400 400
            #生成3个进入atten
            #400->200
            x11 = self.re(self.bn1(self.bimap11(x)))
            x12 = self.re(self.bn2(self.bimap12(x)))
            x13 = self.re(self.bn3(self.bimap13(x)))
            x1 = torch.stack([x11,x12,x13]).squeeze(2).permute(1, 0, 2, 3)
            #经过att1变成float了 是不是att参数的问题
            x, shape = self.att1(x1)
            x = (torch.sum(x,dim=1).unsqueeze(1))/3
            x_spd=self.re((self.bimap2(x)))
            # x_spd_new = self.re(self.batchnorm4(self.bimap4(x_spd)))
            # x_spd_new = self.batchnorm5(self.bimap5(x_spd_new))
            #
            # x_spd = x_spd + x_spd_new
            #200->100
            # x_spd=self.re((self.bimap2(x_spd)))

            # x_spd = x_spd + self.re((self.bimap6(x)))

            #100->50
            # x_spd=self.batchnorm3(self.bimap3(x_spd))
            x_vec=self.logeig(x_spd).view(x_spd.shape[0],-1)
            y=self.linear(x_vec)
            return y
    model=AfewNet()

    #setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opti = MixOptimizer(model.parameters(),lr=lr)

    #initial validation accuracy
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
    print('Initial validation accuracy: '+str(100*acc_val)+'%')

    #training loop
    for epoch in range(epochs):

        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        i=0
        for local_batch, local_labels in data_loader._train_generator:
            i=i+1
            time_start = time.time()
            opti.zero_grad()
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1)==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            opti.step()
            time_end = time.time()
            time_c= time_end - time_start
            print("the %d time cost %.2f loss:%.2f  acc:%.2f" % (i,float(time_c), float(loss), float(acc)))
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()
        print("the %d epoch cost loss:%.2f  acc:%.2f" % (epoch, float(loss_train), float(acc_train)))
        writer.add_scalar("train_loss", loss_train, epoch)
        writer.add_scalar("train_acc",acc_train,epoch)
        writer.flush()
        # validation
        loss_val,acc_val=[],[]
        y_true,y_pred=[],[]
        gen = data_loader._test_generator
        model.eval()
        j=0
        for local_batch, local_labels in gen:
            j = j+1
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels=out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_val.append(loss)
            acc_val.append(acc)
            print("the %d  loss:%.2f  acc:%.2f" % (i,float(loss), float(acc)))

        acc_val = np.asarray(acc_val).mean()
        loss_val = np.asarray(loss_val).mean()
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))
        print("the %d epoch  loss:%.2f  acc:%.2f" % (epoch,  float(loss_val), float(acc_val)))

        writer.add_scalar("val_loss", loss_val, epoch)
        writer.add_scalar("val_acc", acc_val, epoch)
        if epoch % 50 == 0:
            th.save(model.state_dict(),'attenepoch'+str(epoch)+'.pt')


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
        def __init__(self,data_path,batch_size):
            path_train,path_test=data_path+'train/',data_path+'val/'
            for filenames in os.walk(path_train):
                names_train = sorted(filenames[2])
            for filenames in os.walk(path_test):
                names_test = sorted(filenames[2])
            self._train_generator=data.DataLoader(DatasetSPD(path_train,names_train),batch_size=batch_size,shuffle='True')
            self._test_generator=data.DataLoader(DatasetSPD(path_test,names_test),batch_size=batch_size,shuffle='False')

    afew(DataLoaderAFEW(data_path,batch_size))
