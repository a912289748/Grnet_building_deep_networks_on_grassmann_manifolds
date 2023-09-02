import math

import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function as F

class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass

def init_bimap_parameter(W):
    """ initializes a (ho,hi,ni,no) 4D-StiefelParameter"""
    ni,no=W.shape
    if ni > no:
        v = th.empty(ni, ni, dtype=W.dtype, device=W.device).uniform_(0., 1.)
        vv = th.svd(v.matmul(v.t()))[0][:, :no]
    else:
        v = th.empty(no, no, dtype=W.dtype, device=W.device).uniform_(0., 1.)
        vv = th.svd(v.matmul(v.t()))[0][:, :ni]
    W.data=vv.t()
    # return
    # if ni > no:
    #     for i in range(ho):
    #         for j in range(hi):
    #             v=th.empty(ni,ni,dtype=W.dtype,device=W.device).uniform_(0.,1.)
    #             vv=th.svd(v.matmul(v.t()))[0][:,:no]
    #             W.data[i,j]=vv
    # else:
    #     for i in range(ho):
    #         for j in range(hi):
    #             v=th.empty(no,no,dtype=W.dtype,device=W.device).uniform_(0.,1.)
    #             vv=th.svd(v.matmul(v.t()))[0][:,:ni]
    #             W.data[i,j]=vv.T

def init_bimap_parameter_identity(W):
    """ initializes to identity a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho,hi,ni,no=W.shape
    for i in range(ho):
        for j in range(hi):
            W.data[i,j]=th.eye(ni,no)

class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass

def bimap(X,W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    return W.t().matmul(X).matmul(W)

def bimaps(X,W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    return X.matmul(W)

def bimaps_channels(X,W):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    # Pi=th.zeros(X.shape[0],1,W.shape[-1],W.shape[-1],dtype=X.dtype,device=X.device)
    # for j in range(X.shape[1]):
    #     Pi=Pi+bimap(X,W[j])
    batch_size,channels_in,n_in,_=X.shape
    channels_out,_,_,n_out=W.shape
    P=th.zeros(batch_size,channels_out,n_in,n_out,dtype=X.dtype,device=X.device)
    for co in range(channels_out):
        P[:,co,:,:]=sum([bimaps(X[:,ci,:,:],W[co,ci,:,:]) for ci in range(1)])
        #?
        # P[:,co,:,:]=[bimap(X[:,ci,:,:],W[co,ci,:,:]) for ci in range(channels_in)]
    return P

def bimap_channels(X,W):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    # Pi=th.zeros(X.shape[0],1,W.shape[-1],W.shape[-1],dtype=X.dtype,device=X.device)
    # for j in range(X.shape[1]):
    #     Pi=Pi+bimap(X,W[j])
    batch_size,channels_in,n_in,_=X.shape
    channels_out,_,_,n_out=W.shape
    P=th.zeros(batch_size,channels_out,n_out,n_out,dtype=X.dtype,device=X.device)
    for co in range(channels_out):
        P[:,co,:,:]=sum([bimap(X[:,ci,:,:],W[co,ci,:,:]) for ci in range(channels_in)])
        #?
        # P[:,co,:,:]=[bimap(X[:,ci,:,:],W[co,ci,:,:]) for ci in range(channels_in)]
    return P

def modeig_forward(P,op,eig_mode='svd',param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size,channels,n,n=P.shape #batch size,channel depth,dimension
    U,S=th.zeros_like(P,device=P.device),th.zeros(batch_size,channels,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if(eig_mode=='eig'):
                s,U[i,j]=th.eig(P[i,j],True); S[i,j]=s[:,0]
                # 改动
                # s,U[i,j]=th.linalg.eig(P[i,j]); S[i,j]=s[:,0]
            elif(eig_mode=='svd'):
                #这里打断点
                U[i,j],S[i,j],_=th.svd(P[i,j])
    S_fn=op.fn(S,param)
    X=U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2,3))
    # X = X[..., :X.shape[-1], :][..., torch.triu(torch.ones(X.shape[-1], X.shape[-1])) == 1]
    return X,U,S,S_fn

def modeig_forward_log(P,op,eig_mode='svd',param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size,channels,n,n=P.shape #batch size,channel depth,dimension
    U,S=th.zeros_like(P,device=P.device),th.zeros(batch_size,channels,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            if(eig_mode=='eig'):
                s,U[i,j]=th.eig(P[i,j],True); S[i,j]=s[:,0]
                # 改动
                # s,U[i,j]=th.linalg.eig(P[i,j]); S[i,j]=s[:,0]
            elif(eig_mode=='svd'):
                #这里打断点
                U[i,j],S[i,j],_=th.svd(P[i,j])
    S_fn=op.fn(S,param)
    X=U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2,3))
    X = X[..., :X.shape[-1], :][..., torch.triu(torch.ones(X.shape[-1], X.shape[-1])) == 1]
    return X,U,S,S_fn

def modeig_backward_log(dx,U,S,S_fn,op,param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    record_dim = dx.shape[1]
    if dx.shape[1] != 1:
        S = S.reshape(S.shape[0] * S.shape[1], 1, -1)
        S_fn = S_fn.reshape(S_fn.shape[0] * S_fn.shape[1], 1, -1)
        U = U.reshape(U.shape[0]*U.shape[1],1,U.shape[-1],U.shape[-2])
        S = S.reshape(S.shape[0]*S.shape[1],1,S.shape[-1])

    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    S_fn_deriv=BatchDiag(op.fn_deriv(S,param))
    SS=S[...,None].repeat(1,1,1,S.shape[-1])#
    SS_fn=S_fn[...,None].repeat(1,1,1,S_fn.shape[-1])
    L=(SS_fn-SS_fn.transpose(2,3))/(SS-SS.transpose(2,3))
    L[L==-np.inf]=0; L[L==np.inf]=0; L[th.isnan(L)]=0
    L=L+S_fn_deriv

    n = int( (-1 + math.sqrt(1 + 8*dx.shape[2])) / 2)

    symmetric_tensor = torch.zeros((dx.shape[0]*dx.shape[1], n, n))

    dx = dx.reshape(dx.shape[0] * dx.shape[1], 1, -1)
    indices = torch.triu_indices(n, n, offset=0)
    symmetric_tensor[:, indices[0], indices[1]] = dx.squeeze()
    symmetric_tensor = symmetric_tensor + torch.transpose(symmetric_tensor, 1, 2)
    symmetric_tensor = symmetric_tensor.unsqueeze(1)
    dp=L*(U.transpose(2,3).matmul(symmetric_tensor).matmul(U))
    dp=U.matmul(dp).matmul(U.transpose(2,3))
    return dp.reshape(30,record_dim,dp.shape[-1],dp.shape[-2])

def modeig_backward(dx,U,S,S_fn,op,param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    S_fn_deriv=BatchDiag(op.fn_deriv(S,param))
    SS=S[...,None].repeat(1,1,1,S.shape[-1])#
    SS_fn=S_fn[...,None].repeat(1,1,1,S_fn.shape[-1])
    L=(SS_fn-SS_fn.transpose(2,3))/(SS-SS.transpose(2,3))
    L[L==-np.inf]=0; L[L==np.inf]=0; L[th.isnan(L)]=0
    L=L+S_fn_deriv
    dp=L*(U.transpose(2,3).matmul(dx).matmul(U))
    dp=U.matmul(dp).matmul(U.transpose(2,3))
    return dp



class SubLogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward_log(P,Log_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward_log(dx,U,S,S_fn,Log_op)

class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Log_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Log_op)
class QRs(F):
    @staticmethod
    def forward(ctx,P):


        [Q,R] = torch.qr(P)
        # n1,n2,w1,w2 = P.shape
        # for i in range(n1):
        #     for j in range(n2):
        #         P[i][j] = 1
        ctx.save_for_backward(Q,R)
        return Q

    @staticmethod
    def backward(ctx,dx):
        [Qs,Rs] = ctx.saved_variables
        n1,n2,w1,w2 = dx.shape
        Y = torch.zeros(n1,n2,w2,w1)
        for i in range(n1):
            for j in range(n2):
                T = dx[i][j]
                Q = Qs[i][j]
                R = Rs[i][j]

                m = T.shape[0]
                S = torch.eye(m) - torch.matmul(Q, Q.T)
                dzdx_to = torch.matmul(Q.T, T)
                dzdx_t1 = torch.tril(dzdx_to) - torch.diag_embed(torch.diag(dzdx_to))
                dzdx_t2 = torch.tril(dzdx_to.T) - torch.diag_embed(torch.diag(dzdx_to.T))

                dzdx = (torch.matmul(S.T, T) + torch.matmul(torch.matmul(Q, (dzdx_t1 - dzdx_t2)), torch.inverse(R))).T
                # dzdx = (torch.matmul(S.T, T) + torch.matmul(torch.matmul(Q, (dzdx_t1 - dzdx_t2)), torch.inverse(R+torch.diag_embed(torch.trace(R).repeat(1,R.shape[0]))))).T
                Y[i][j] = dzdx
                # dzdx = (S'*dLdQ+Q*(dzdx_t1-dzdx_t2))*(inv(R))';

        # Q = ctx.saved_variables[1]
        # R = ctx.saved_variables[2]

        return Y.transpose(2,3)
class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Re_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Re_op)


class Orthmap(F):
    @staticmethod
    def forward(ctx, X):

        U,S,V = torch.svd(X)
        ctx.save_for_backward(X,U,S)

        return U[:, :, :, :10]

    @staticmethod
    def calculate_grad_svd(U, S, p, D, dzdy):
        pass
    @staticmethod
    def backward(ctx,P):
        X = ctx.saved_variables[0]
        U = ctx.saved_variables[1]
        S = ctx.saved_variables[2]
        n1,n2,w,h =  P.shape
        n11,n22,w1,h1 = X.shape
        Y = torch.zeros(n11,n22,w1,h1)
        for i3 in range(n1):
            for i4 in range(n2):
                U_t = U[i3,i4,:,:]
                S_t = torch.diag(S[i3,i4,:])

                D = S_t.shape[0]
                p = 10
                dzdy = P[i3,i4,:,:]


                diagS = torch.diag(S_t)
                Dmin = len(diagS)
                ind = torch.arange(1, Dmin + 1)
                dLdC = torch.zeros(D, D).double()
                A = torch.zeros(D,D)
                A[:, :p] = 1

                # dLdC[A == 1] = dzdy
                dLdC[:D,:p]=dzdy
                dLdU =dLdC.clone()
                if sum(ind) == 1:
                    pass
                else:
                    e = diagS
                    dim = e.size(0)
                    s = e.view(dim, 1)
                    ones = torch.ones(1, dim, dtype=torch.float64)
                    s = s @ ones
                    k = 1 / (s - s.t())
                    k[torch.eye(dim) > 0] = 0
                    k[k == float("Inf")] = 0
                    indices = (diagS < 1e-10).nonzero().squeeze()
                    if len(indices) != 0:
                        k[indices, indices] = 0
                    ss= k.t().mul(U_t.t()@dLdU)
                    ss = (ss + ss.t()) / 2
                    Y[i3,i4,:,:]=U_t@ss@U_t.t()
                # Y[i3,i4,:,:]=calculate_grad_svd(U_t,S_t,10,S_t.shape[1],P[i3,i4,:,:])
        # print(n2)
        # print(w)
        # print(h)
        # dzdy = torch.bmm(P.view(-1,w,h),X.view(-1,w1,h1)).view(n1,n2,w,h1)

        return Y

    @staticmethod
    def calculate_grad_svd( U, S, p, D, dzdy):
        pass
class ProjMap(F):
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        ch1, ch2, w, h = X.shape
        P1 = X.view(-1, w, h)
        P2 = torch.bmm(P1, P1.transpose(1, 2))
        # n1, n2, w1, h1 = P2.shape
        # print(P2.shape)
        P2 = P2.view(30, 8, w, w)
        return P2
    @staticmethod
    def backward(ctx,P):
        X = ctx.saved_variables[0]
        n1,n2,w,h =  P.shape
        n11,n22,w1,h1 = X.shape
        res = torch.zeros(n1,n2,w,h1).double()
        #P = d_t
        for ix in range(n1):
            d_t = P[ix,:,:,:]
            for iy in range(n2):
                res[ix,iy,:,:]=2.*(torch.matmul(d_t[iy,:,:],X[ix,iy,:,:]))
        # print(n2)
        # print(w)
        # print(h)

        return res

class SubConv(F):
    @staticmethod
    def forward(ctx, X,k):
        # ctx.save_for_backward(X)
        ch1, ch2, w, h = X.shape
        P1 = X.view(-1, w, h)
        # Get input matrix dimensions
        stride = 1
        imagesize = int(w** 0.5)
        outsize = (imagesize - k) // stride + 1

        # Check input matrix dimensions
        if imagesize != int(imagesize) or outsize != int(outsize):
            raise ValueError("Wrong size of X")
        if k == 2:
            outsize = 4
        elif k == 3:
            outsize = 9
        # Initialize output matrices
        Y = torch.zeros( (imagesize - k + 1)**2,30,k**2,k**2)
        idx = torch.zeros(outsize ** 2, k ** 2, dtype=torch.int64)
        n = 0

        # Loop over sliding windows
        for ir in range(1, imagesize - k + 2):
            for ic in range( 1, imagesize - k + 2):
                # ic = ic+1
                # ir = ir +1
                temp_idx = torch.zeros((k, k), dtype=torch.int64)

                # Construct index matrix for current sliding window
                for ii in range(k):

                    col_start = imagesize * (ic - 1) + ir + imagesize * (ii )
                    col = torch.linspace(col_start, col_start + k - 1, k, dtype=torch.int64)

                    temp_idx[ii:, ] = col

                # Flatten current sliding window and save to output matrices
                temp = temp_idx.flatten()
                temp = temp-1
                idx[n, :] = temp
                # torch.index_select(torch.index_select(X[0][0], 0, temp),1,temp)
                temp_Y = torch.index_select(X, 2, temp)
                temp_Y = torch.index_select(temp_Y, 3, temp)
                # Y=temp_Y.view(-1, 16) 这个不必要按照matlab 因为matlab还原了
                Y[n] = temp_Y.squeeze(1)
                n += 1
        # P2 = torch.bmm(P1, P1.transpose(1, 2))
        # n1, n2, w1, h1 = P2.shape
        # print(P2.shape)
        ctx.save_for_backward(X,idx) #保存索引，后面恢复的时候用

        return Y.permute(1,0,2,3)
    @staticmethod
    def backward(ctx,P):
        X,idx = ctx.saved_variables
        # n1,w,h =  P.shape
        # n11,n22,w1,h1 = X.shape
        n1,ch1,w,h = P.shape
        dzdy = torch.zeros(n1,1,25,25)
        # # print(n2)
        # # print(w)
        # # print(h)
        for i in range(n1):
            for j in range(ch1):
                temp_idx = idx[j, :]
                dzdy[i][0][temp_idx[:, None], temp_idx] =dzdy[i][0][temp_idx[:, None], temp_idx] +  P[i][j]
        # dzdy = torch.bmm(P,X.view(-1,w1,h1))
        # dzdy = dzdy.unsqueeze(0)
        return dzdy,None

class ProjMapSPD(F):
    @staticmethod
    def forward(ctx, X):
        ctx.save_for_backward(X)
        ch1, ch2, w, h = X.shape
        P1 = X.view(-1, w, h)
        P2 = torch.bmm(P1, P1.transpose(1, 2))
        # n1, n2, w1, h1 = P2.shape
        # print(P2.shape)
        return P2
    @staticmethod
    def backward(ctx,P):
        X = ctx.saved_variables[0]
        n1,w,h =  P.shape
        n11,n22,w1,h1 = X.shape

        # print(n2)
        # print(w)
        # print(h)
        dzdy = torch.bmm(P,X.view(-1,w1,h1))
        dzdy = dzdy.unsqueeze(0)
        return dzdy

class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Exp_op,eig_mode='eig')
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Exp_op)

class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqm_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqm_op)

class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Sqminv_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Sqminv_op)

class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P,power):
        Power_op._power=power
        X,U,S,S_fn=modeig_forward(P,Power_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Power_op),None

class InvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse eigenvalues matrices of size (n,n)
    """
    @staticmethod
    def forward(ctx,P):
        X,U,S,S_fn=modeig_forward(P,Inv_op)
        ctx.save_for_backward(U,S,S_fn)
        return X
    @staticmethod
    def backward(ctx,dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U,S,S_fn=ctx.saved_variables
        return modeig_backward(dx,U,S,S_fn,Inv_op)

def geodesic(A,B,t):
    '''
    Geodesic from A to B at step t
    :param A: SPD matrix (n,n) to start from
    :param B: SPD matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (n,n) along the geodesic
    '''
    M=CongrG(PowerEig.apply(CongrG(B,A,'neg'),t),A,'pos')[0,0]
    return M

def cov_pool(f,reg_mode='mle'):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    bs,n,T=f.shape
    X=f.matmul(f.transpose(-1,-2))/(T-1)
    if(reg_mode=='mle'):
        ret=X
    elif(reg_mode=='add_id'):
        ret=add_id(X,1e-6)
    elif(reg_mode=='adjust_eig'):
        ret=adjust_eig(X,0.75)
    if(len(ret.shape)==3):
        return ret[:,None,:,:]
    return ret

def cov_pool_mu(f,reg_mode):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    alpha=1
    bs,n,T=f.shape
    mu=f.mean(-1,True); f=f-mu
    X=f.matmul(f.transpose(-1,-2))/(T-1)+alpha*mu.matmul(mu.transpose(-1,-2))
    aug1=th.cat((X,alpha*mu),2)
    aug2=th.cat((alpha*mu.transpose(1,2),th.ones(mu.shape[0],1,1,dtype=mu.dtype,device=f.device)),2)
    X=th.cat((aug1,aug2),1)
    if(reg_mode=='mle'):
        ret=X
    elif(reg_mode=='add_id'):
        ret=add_id(X,1e-6)
    elif(reg_mode=='adjust_eig'):
        ret=adjust_eig(0.75)(X)
    if(len(ret.shape)==3):
        return ret[:,None,:,:]
    return ret

def add_id(P,alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    for i in range(P.shape[0]):
        P[i]=P[i]+alpha*P[i].trace()*th.eye(P[i].shape[-1],dtype=P.dtype,device=P.device)
    return P

def dist_riemann(x,y):
    '''
    Riemannian distance between SPD matrices x and SPD matrix y
    :param x: batch of SPD matrices (batch_size,1,n,n)
    :param y: single SPD matrix (n,n)
    :return:
    '''
    return LogEig.apply(CongrG(x,y,'neg')).view(x.shape[0],x.shape[1],-1).norm(p=2,dim=-1)

def CongrG(P,G,mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if(mode=='pos'):
        GG=SqmEig.apply(G[None,None,:,:])
    elif(mode=='neg'):
        GG=SqminvEig.apply(G[None,None,:,:])
    # 转换格式
    # P = torch.Tensor(P,dtype=torch.float64)
    # GG =

    PP=GG.matmul(P).matmul(GG)
    return PP

def LogG(x,X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x,X,'neg')),X,'pos')

def LogGGrass(x,X):
    _,_,n,d = x.shape
    # 对S的列向量进行Gram-Schmidt正交化
    res = torch.zeros_like(x)
    for i in range(x.shape[0]):
        G1 = x[i][0]
        G2 = X
        # proj = G1 - G2 @ torch.inverse((G2.T @ G2)) @ G2.T @ G1
        GT2G2_inv = torch.inverse(torch.matmul(torch.transpose(G2, 0, 1), G2))
        proj = torch.sub(G1, torch.matmul(G2, torch.matmul(GT2G2_inv, torch.matmul(torch.transpose(G2, 0, 1), G1))))
        res[i, 0, :, :] = proj

    return res

def ExpG(x,X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x,X,'neg')),X,'pos')

def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size,channels,n=P.shape #batch size,channel depth,dimension
    Q=th.zeros(batch_size,channels,n,n,dtype=P.dtype,device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i,j]=P[i,j].diag()
    return Q

def karcher_step(x,G,alpha):
    '''
    One step in the Karcher flow
    '''
    x_log=LogG(x,G)
    G_tan=x_log.mean(dim=0)[None,...]
    G=ExpG(alpha*G_tan,G)[0,0]
    return G

def BaryGeomGrass(x):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k=1
    alpha=1
    with th.no_grad():
        G=th.mean(x,dim=0)[0,:,:] #批次的均值 200 200 按照batch方向求均值
        for _ in range(k):
            G=karcher_grass_step(x,G,alpha)
        return G

def karcher_grass_step(x,G,alpha):
    '''
    One step in the Karcher flow
    '''
    #240个矩阵 投影到一个矩阵上
    x_log=LogGGrass(x,G)
    G_tan=x_log.mean(dim=0)[None,...]
    G=ExpG(alpha*G_tan,G)[0,0]
    return G

def BaryGeomGrass(x):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k=1
    alpha=1
    with th.no_grad():
        G=th.mean(x,dim=0)[0,:,:] #批次的均值 200 200 按照batch方向求均值
        G = torch.qr(G)[0]
        # G = th.mean(x.view(-1,400,200),dim=0)
        for _ in range(k):
            G=karcher_grass_step(x,G,alpha)
        return G
def BaryGeom(x):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k=1
    alpha=1 #通道数为8 可以看做
    with th.no_grad():
        G=th.mean(x,dim=0)[0,:,:] #批次的均值 200 200 按照batch方向求均值
        for _ in range(k):
            G=karcher_step(x,G,alpha)
        return G

def karcher_step_weighted(x,G,alpha,weights):
    '''
    One step in the Karcher flow
    Weights is a weight vector of shape (batch_size,)
    Output is mean of shape (n,n)
    '''
    x_log=LogG(x,G)
    G_tan=x_log.mul(weights[:,None,None,None]).sum(dim=0)[None,...]
    G=ExpG(alpha*G_tan,G)[0,0]
    return G
def bary_geom_weighted(x,weights):
    '''
    Function which computes the weighted Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Weights is a weight vector of shape (batch_size,)
    Output is (1,1,n,n) Riemannian mean
    '''
    k=1
    alpha=1
    # with th.no_grad():
    G=x.mul(weights[:,None,None,None]).sum(dim=0)[0,:,:]
    for _ in range(k):
        G=karcher_step_weighted(x,G,alpha,weights)
    return G[None,None,:,:]

class Log_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.log(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 1/S

class Re_op():
    """ Log function and its derivative """
    _threshold=1e-4
    @classmethod
    def fn(cls,S,param=None):
        return nn.Threshold(cls._threshold,cls._threshold)(S)
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (S>cls._threshold).double()

class Sqm_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return 0.5/th.sqrt(S)

class Sqminv_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return 1/th.sqrt(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return -0.5/th.sqrt(S)**3

class Power_op():
    """ Power function and its derivative """
    _power=1
    @classmethod
    def fn(cls,S,param=None):
        return S**cls._power
    @classmethod
    def fn_deriv(cls,S,param=None):
        return (cls._power)*S**(cls._power-1)

class Inv_op():
    """ Inverse function and its derivative """
    @classmethod
    def fn(cls,S,param=None):
        return 1/S
    @classmethod
    def fn_deriv(cls,S,param=None):
        return log(S)

class Exp_op():
    """ Log function and its derivative """
    @staticmethod
    def fn(S,param=None):
        return th.exp(S)
    @staticmethod
    def fn_deriv(S,param=None):
        return th.exp(S)