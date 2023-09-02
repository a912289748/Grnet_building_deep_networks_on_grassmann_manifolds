import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function as F
from . import functional

dtype=th.double
device=th.device('cpu')



class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """
    def __init__(self,ho,hi,ni,no):
        super(BiMap, self).__init__()
        #19个
        self._W=functional.StiefelParameter(th.empty(ho,hi,ni,no,dtype=dtype,device=device))
        self._ho=ho; self._hi=hi; self._ni=ni; self._no=no
        functional.init_bimap_parameter(self._W)
        # print(self._W.grad)
        # print("hello")
    def forward(self,X):
        # print("forward"+str(self._W.grad!=None))

        return functional.bimap_channels(X,self._W)#the weight is 1 1 400 200


class BiMaps(nn.Module):
    def __init__(self,ho,hi,ni,no):
        super(BiMaps, self).__init__()
        self._W1 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._W2 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._W3 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._W4 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._W5 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._W6 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._W7 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._W8 = nn.Parameter(th.empty(ni,no,dtype=dtype,device=device))
        self._ho=ho; self._hi=hi; self._ni=ni; self._no=no
        functional.init_bimap_parameter(self._W1)
        functional.init_bimap_parameter(self._W2)
        functional.init_bimap_parameter(self._W3)
        functional.init_bimap_parameter(self._W4)
        functional.init_bimap_parameter(self._W5)
        functional.init_bimap_parameter(self._W6)
        functional.init_bimap_parameter(self._W7)
        functional.init_bimap_parameter(self._W8)

    def forward(self,X):
        # ch1,ch2,w,h = X.shape
        X = X.double()
        if len(X.shape) == 3:
            # W1 = torch.matmul(X,self._W1)
            # W2 = torch.matmul(X,self._W2)
            # W3 = torch.matmul(X,self._W3)
            # W4 = torch.matmul(X,self._W1)
            # W5 = torch.matmul(X,self._W5)
            # W6 = torch.matmul(X,self._W6)
            # W7 = torch.matmul(X,self._W7)
            # W8 = torch.matmul(X,self._W8)

            w1 = torch.matmul(self._W1.unsqueeze(0), X)
            w2 = torch.matmul(self._W2.unsqueeze(0), X)
            w3 = torch.matmul(self._W3.unsqueeze(0), X)
            w4 = torch.matmul(self._W4.unsqueeze(0), X)
            w5 = torch.matmul(self._W5.unsqueeze(0), X)
            w6 = torch.matmul(self._W6.unsqueeze(0), X)
            w7 = torch.matmul(self._W7.unsqueeze(0), X)
            w8 = torch.matmul(self._W8.unsqueeze(0), X)
            return th.stack([w1,w2,w3,w4,w5,w6,w7,w8],dim=1)

        else:

            res = torch.empty((X.shape[0],X.shape[1],self._W1.shape[0],X[:,0,:,:].shape[2])).double()
            res[:, 0, :, :] = torch.matmul(self._W1.unsqueeze(0),X[:,0,:,:])
            res[:, 1, :, :] = torch.matmul(self._W2.unsqueeze(0),X[:,1,:,:])
            res[:, 2, :, :] = torch.matmul(self._W3.unsqueeze(0),X[:,2,:,:])
            res[:, 3, :, :] = torch.matmul(self._W4.unsqueeze(0),X[:,3,:,:])
            res[:, 4, :, :] = torch.matmul(self._W5.unsqueeze(0),X[:,4,:,:])
            res[:, 5, :, :] = torch.matmul(self._W6.unsqueeze(0),X[:,5,:,:])
            res[:, 6, :, :] = torch.matmul(self._W7.unsqueeze(0),X[:,6,:,:])
            res[:, 7, :, :] = torch.matmul(self._W8.unsqueeze(0),X[:,7,:,:])



            return res
        #_W 400 200  X是400 400+++

class SubLogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.SubLogEig.apply(P)

class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.LogEig.apply(P)


class SubConv(nn.Module):
    def __init__(self,k):
        super(__class__,self).__init__()
        self.k = k
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    def forward(self,P,k):
        return functional.SubConv.apply(P,k)
class ExpEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.ExpEig.apply(P)
class SqmEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of sqrt eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.SqmEig.apply(P)

class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    def forward(self,P):
        return functional.ReEig.apply(P)

class Qrs(nn.Module):
    def forward(self,x):
        return functional.QRs.apply(x)

class Orthmap(nn.Module):
    def forward(self,x):
        return functional.Orthmap.apply(x)
class Projmap(nn.Module):
    def forward(self,x):
        return functional.ProjMap.apply(x)

class ProjmapSPD(nn.Module):
    def forward(self,x):
        return functional.ProjMapSPD.apply(x)
class BaryGeom(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    def forward(self,x):
        return functional.BaryGeom(x)

class BatchNormGrass(nn.Module):
    def __init__(self, n,s):
        super(__class__, self).__init__()
        pass
    def forward(self,X):
        N, h, n, ns = X.shape
        # 不管多少个channel 合并成一个channel 相当于打包，对应后面有解包，相当于对channel的压缩分开操作 主要方便下面的处理
        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, ns, N * h, 1).permute(2, 3, 0, 1).contiguous()
        mean = functional.BaryGeomGrass(X_batched)

        if (self.training):
            mean = functional.BaryGeom(X)
            with th.no_grad():
                pass
        else:
            return
class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized cdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddcdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
    SPD parameter of size (n,n)
    """
    def __init__(self,n):
        super(__class__,self).__init__()
        self.momentum=0.1
        self.running_mean=th.eye(n,dtype=dtype) ################################
        # self.running_mean=nn.Parameter(th.eye(n,dtype=dtype),requires_grad=False)
        self.weight=functional.SPDParameter(th.eye(n,dtype=dtype))
    def forward(self,X):
        N,h,n,n=X.shape
        #不管多少个channel 合并成一个channel 相当于打包，对应后面有解包，相当于对channel的压缩分开操作 主要方便下面的处理
        X_batched=X.permute(2,3,0,1).contiguous().view(n,n,N*h,1).permute(2,3,0,1).contiguous()
        if(self.training):
            mean=functional.BaryGeom(X_batched)
            with th.no_grad():

                self.running_mean.data=functional.geodesic(self.running_mean,mean,self.momentum)
            X_centered=functional.CongrG(X_batched,mean,'neg')
        else:
            X_centered=functional.CongrG(X_batched,self.running_mean,'neg')
        X_normalized=functional.CongrG(X_centered,self.weight,'pos')
        return X_normalized.permute(2,3,0,1).contiguous().view(n,n,N,h).permute(2,3,0,1).contiguous()

class CovPool(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """
    def __init__(self,reg_mode='mle'):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode
    def forward(self,f):
        return functional.cov_pool(f,self._reg_mode)

class CovPoolBlock(nn.Module):
    """
    Input f: L blocks of temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,L,n,T)
    Output X: L covariance matrices, shape (batch_size,L,1,n,n)
    """
    def __init__(self,reg_mode='mle'):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode
    def forward(self,f):
        ff=[functional.cov_pool(f[:,i,:,:],self._reg_mode)[:,None,:,:,:] for i in range(f.shape[1])]
        return th.cat(ff,1)

class CovPoolMean(nn.Module):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output X: Covariance matrix of size (batch_size,1,n,n)
    """
    def __init__(self,reg_mode='mle'):
        super(__class__,self).__init__()
        self._reg_mode=reg_mode
    def forward(self,f):
        return functional.cov_pool_mu(f,self._reg_mode)