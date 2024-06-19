import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from EGNN import EGNN
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# t-SNE 
tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000)


def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

class Subgraphnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act=F.hardtanh, drop_p=0.0):
        super(Subgraphnet, self).__init__()
        self.ks = ks
      
        self.act = act
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.subgraph_gcns_1 = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)
        self.l_n = len(ks)
        self.drop_p = drop_p

        for j in range(3):

            self.subgraph_gcns_1.append(EGNN(dim = in_dim,edge_dim=1, m_dim =17))# ,dropout=self.drop_p)) # 16 20
  
        for i in range(self.l_n):
           
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def readout(self, hs):
        h_max = torch.Tensor([torch.max(h, 0)[0] for h in hs])

        return h_max

    def forward(self, feat, coor, edge , ep):
    
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = feat[0,:,:]
        org_g = edge[0,:,:,0]
        org_c = coor[0,:,:]



        
        for i in range(self.l_n):
            g = org_g
            h = org_h
            c = org_c
            if self.ks[i] != 1:
                g, h, idx = self.pools[i](g, h, ep)
                c = c[idx,:]

       
            
            # egnn
            feat1 = h.unsqueeze(0)
            edge1 = g.unsqueeze(0)
            edge1 = edge1.unsqueeze(edge1.dim())
            coor1 = c.unsqueeze(0)
            h = feat1

            for j in range(3):
               

                h_temp, coor1 =  self.subgraph_gcns_1[j](h, coor1, edge1,ep)
                h = torch.relu(h + h_temp)


            h = h[0,:,:]

            if self.ks[i] != 1:
                g, h = self.unpools[i](org_g, h, org_h, idx)

            hs.append(h)
        
        
     
        
        for h_i in hs:
            h = torch.max(h,h_i)
 
        return h



    
class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act=F.hardtanh, drop_p=0.0):
        super(GraphUnet, self).__init__()
        self.ks = ks
      

        self.bottom_gcn = EGNN(dim = in_dim,edge_dim=1)

        self.act = act
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.subgraph_gcns_1 = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)
        self.l_n = len(ks)

        self.down_gcns_1 = nn.ModuleList()
        self.down_gcns_2 = nn.ModuleList()
        self.down_gcns_3 = nn.ModuleList()


        self.up_gcns_1 = nn.ModuleList()
        self.up_gcns_2 = nn.ModuleList()
        self.up_gcns_3 = nn.ModuleList()


        self.bottom_gcns_1 = EGNN(dim = in_dim,edge_dim=1)




        for i in range(self.l_n):

            # egnn
            self.down_gcns_1.append(EGNN(dim = in_dim,edge_dim=1))
            self.down_gcns_2.append(EGNN(dim = in_dim,edge_dim=1))
            self.down_gcns_3.append(EGNN(dim = in_dim,edge_dim=1))

            self.up_gcns_1.append(EGNN(dim = in_dim,edge_dim=1))
            self.up_gcns_2.append(EGNN(dim = in_dim,edge_dim=1))
            self.up_gcns_3.append(EGNN(dim = in_dim,edge_dim=1))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def readout(self, hs):

        h_sum = [torch.sum(h, 0) for h in hs]
        h_mean = [torch.mean(h, 0) for h in hs]
        h = torch.cat(h_max + h_sum + h_mean)
        return h

    

  

    def forward(self, feat, coor, edge , ep): 

        adj_ms = []
        coor_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = feat
        org_c = coor
        org_e = edge

        for i in range(self.l_n):
            h = feat + self.down_gcns_1[i](feat, coor, edge,ep)
            h = torch.relu(h)
            h = self.down_gcns_2[i](h, coor, edge,ep)
            h = torch.relu(h)
            h = self.down_gcns_3[i](h, coor, edge,ep)
            h = torch.relu(h)
           

            g = edge[0,:,:,0]
            h = h[0,:,:]
            coor_ms.append(coor)
            c = coor[0,:,:]
            adj_ms.append(g)
            down_outs.append(h)

           

            g, h, idx = self.pools[i](g, h, ep)

            feat = h.unsqueeze(0)
            edge = g.unsqueeze(0)
            edge = edge.unsqueeze(edge.dim())

            c = c[idx,:]
            coor = c.unsqueeze(0)
            indices_list.append(idx)
        h = feat + self.bottom_gcns_1(feat, coor, edge, ep)
        h = torch.relu(h)

        h = h[0,:,:]
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)

            feat = h.unsqueeze(0)
            edge = g.unsqueeze(0)
            edge = edge.unsqueeze(edge.dim())
            coor = coor_ms[up_idx]

            h = feat
            h = h + self.up_gcns_1[i](feat, coor, edge, ep)
            h = torch.relu(h)
            h = self.up_gcns_2[i](h, coor, edge, ep)
            h = torch.relu(h)
            h = self.up_gcns_3[i](h, coor, edge, ep)
            h = torch.relu(h)



            h = h[0,:,:]
            h = h.add(down_outs[up_idx])


        return h






class Pool(nn.Module):

    def __init__(self, k, in_dim,p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h, ep):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        
        return top_k_graph(scores, g, h, self.k,ep)

class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k, ep):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)#

    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()#
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = un_g
    return g, new_h, idx




