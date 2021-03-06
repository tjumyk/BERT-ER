import torch.nn as nn
import torch
from torch.nn.init import *
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn import Parameter
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def print_matrix(t):
    l_qk = t.tolist()
    for x in l_qk:
        for y in x:
            print(y)

def attention(query, key, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    #print_matrix(mask)
    scores_qk = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores_qk = scores_qk.masked_fill(mask==0, -1e9)
    #print_matrix(scores_qk)
    scores_kq = scores_qk.transpose(-1, -2)
    qk_attn = F.softmax(scores_qk, dim=-1)
    kq_attn = F.softmax(scores_kq, dim=-1)
    if dropout is not None:
        qk_attn = dropout(qk_attn)
        kq_attn = dropout(kq_attn)
    return qk_attn, kq_attn


#designed for cross encoding, the main modification is that we do not do linear transformer on value and the final output
class Cross_Encoding(nn.Module):
    def __init__(self, d_model, dropout=None):
        super(Cross_Encoding, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, A, mask=None):

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = [l(x) for l, x in zip(self.linears, (Q, A))]

        # 2) Apply attention on all the projected vectors in batch.
        qk_attn, kq_attn = attention(query, key,  mask=mask, dropout=self.dropout)
        eq = torch.matmul(qk_attn, A)
        ea = torch.matmul(kq_attn, Q)
        return eq, ea

class Orthogonal_matrix(nn.Module):
    def __init__(self, m, k):
        super(Orthogonal_matrix, self).__init__()
        self.u = torch.zeros([m, k], dtype=torch.float, requires_grad=True).to('cuda')
        nn.init.xavier_uniform_(self.u, gain=1)
        self.s = torch.tensor([1 for _ in range(k)]).to('cuda')
        self.v = torch.zeros([k, k], dtype=torch.float, requires_grad=True).to('cuda')
        nn.init.xavier_uniform_(self.v, gain=1)
        self.m = torch.zeros([m, k], dtype=torch.float, requires_grad=True).to('cuda')
        nn.init.xavier_uniform_(self.m, gain=1)
        #self.warm_up = 300
        self.us = torch.zeros([m, k], dtype=torch.float, requires_grad=True).to('cuda')
        nn.init.xavier_uniform_(self.us, gain=1)

    def forward(self):
        #mi = torch.mm(self.u, torch.diag(self.s).float())
        #self.m = torch.mm(mi, self.v)
        #self.u, self.s, self.v = torch.svd(self.m)
        #self.u = Variable(self.m, requires_grad=True)
        self.m = torch.mm(self.us, self.v)
        u, s, self.v = torch.svd(self.m)
        self.us = torch.mm(u, torch.diag(s).float())
        self.us = Variable(self.us, requires_grad=True)
        return self.us
