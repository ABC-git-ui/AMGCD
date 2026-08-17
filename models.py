import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch.nn.init as init
from torch_geometric.utils import get_laplacian, remove_self_loops, add_self_loops
from scipy.special import comb
from sklearn.cluster import DBSCAN
from itertools import combinations,permutations,chain
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
    
from typing import Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.data import Data
import networkx as nx
import math
import os
import random
import time
import dgl
import dgl.function as fn
import numpy as np
import scipy
from dgl.sampling import pack_traces, random_walk
from torch.utils.data import DataLoader
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, to_undirected, to_networkx
import pandas as pd
import numpy as np
from torch_geometric.utils import  to_scipy_sparse_matrix, degree, from_networkx, to_networkx, remove_self_loops, add_self_loops, coalesce, contains_isolated_nodes, subgraph, k_hop_subgraph
from torch_cluster import graclus_cluster
class mlp(nn.Module):
    def __init__(self, n_representation, hidden_dims):
        super(mlp, self).__init__()
        self.n_representation = n_representation
        self.linear1 = nn.Linear(self.n_representation, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(self, x):
      
        x = F.relu(self.linear1(x)) # N * hidden1_dim
        x = self.dropout(x)
        x = F.relu(self.linear2(x)) # N * hidden2_dim
        x = self.dropout(x)
        x = self.linear3(x) # N * 2
        x = self.sigmoid(x) # N * ( probility of each event )=
        return x

    def init_weights(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)


##backbone graph
def find_diameter_path(T):
    random_paths = nx.shortest_path(T, source=list(T.nodes)[0])
    source = max(random_paths, key=lambda i: len(random_paths[i]))
    source_paths = nx.shortest_path(T, source=source)
    target = max(source_paths, key=lambda i: len(source_paths[i]))
    diameter_path = source_paths[target]

    return diameter_path


def find_trunk(edge_index, max_level):
    trunk_list = list()
    pattern = list()
    graph_data = Data(edge_index=edge_index)
    T = to_networkx(graph_data, to_undirected=True, remove_self_loops=True)

    level = 0
    while T.nodes:
        level += 1
        if level <= max_level:
            level_list = list()  

        isolated_nodes = list(nx.isolates(T))
        if level == 1 and len(isolated_nodes):
            level_list.append(isolated_nodes)
            T.remove_nodes_from(isolated_nodes)

        for c in list(nx.connected_components(T)):
            pattern.append(c)
            S = T.subgraph(c).copy()
            assert len(S.edges) != 0

            diameter_path = find_diameter_path(S)
            level_list.append(diameter_path)
            d_path_edges = list(zip(diameter_path[:-1], diameter_path[1:]))
            T.remove_edges_from(d_path_edges)
            T.remove_nodes_from(list(nx.isolates(T)))

        if level < max_level:
            trunk_list.append(level_list)
        elif not T.nodes:
            trunk_list.append(level_list)

    assert len(trunk_list) <= max_level

    return trunk_list,pattern


class prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index):
        TEMP=F.relu(self.temp)
       
        edge_index1, norm1 = get_laplacian(edge_index,normalization='sym', dtype=x.dtype, num_nodes=x.size(0))
        #2I-L
        edge_index2, norm2= add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
        	x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
        	tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
        	x=tmp[self.K-i-1]
        	x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
        	for j in range(i):
        		x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

        	out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        return out
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class Net_truck(nn.Module):
    def __init__(self,in_channel,hidden_dim,K):
        super(Net_truck, self).__init__()
        self.lin1 = nn.Linear(in_channel, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.m = nn.BatchNorm1d(hidden_dim)
        self.prop1 = prop(K)
 
        self.dropout = 0.1

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
       
        x = self.prop1(x, edge_index)
        cluster_model = DBSCAN(eps=0.2, min_samples=5)
        cluster_model.fit(x.cpu().detach().numpy())
        level = len(set(cluster_model.labels_))

        a,b = find_trunk(edge_index, level)
        edge = []
        for i in a:
            for j in i:
                edge.extend(list(combinations(j,2)))
        truck_edge = torch.from_numpy(np.array(edge)).T.cuda()

        x = self.prop1(x, truck_edge)

        return x





class ProbAttentionLayer(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_nodes: int,
            heads: int = 8,
            negative_slope: float = 0.2,
            bias: float = 1,
            self_loops: bool = True,
            fill_value: Union[float, Tensor, str] = 'mean',
            bfs_depth=2,
            device='cpu',
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.fill_value = fill_value
        self.num_nodes = num_nodes
        self.temp_lin = Linear(in_channels, heads,
                               bias=False, weight_initializer='glorot')
        self.mlp = Linear(128,1)
      
        self.conf_coef = Parameter(torch.zeros([]))
        self.bias = Parameter(torch.ones(1) * bias)
        self.reset_parameters()


    def reset_parameters(self):
        self.temp_lin.reset_parameters()

    def forward(self, x,edge_index):
        N, H = self.num_nodes, self.heads

        normalized_x = x - torch.min(x, 1, keepdim=True)[0]
        normalized_x /= torch.max(x, 1, keepdim=True)[0] - \
                        torch.min(x, 1, keepdim=True)[0]

        x_sorted = torch.sort(normalized_x, -1)[0]
        temp = self.temp_lin(x_sorted)


        conf = F.softmax(x, dim=1).amax(-1)
        deg = degree(edge_index[0, :], self.num_nodes)
        deg_inverse = 1 / deg
        deg_inverse[deg_inverse == float('inf')] = 0

        out = self.propagate(edge_index,
                             temp=temp.view(N, H), 
                             alpha=x,
                             conf=conf)
        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inverse.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias 

        prob = nn.Sigmoid()(out.unsqueeze(1))
 
        return prob

    def message(
            self,
            temp_j: Tensor,
            alpha_j: Tensor,
            alpha_i: OptTensor,
            conf_i: Tensor,
            conf_j: Tensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]) -> Tensor:
        """
        alpha_i, alpha_j: [E, H]
        temp_j: [E, H]
        """
        if alpha_i is None:
            print("alphai is none")
        alpha = (alpha_j * alpha_i).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        # Agreement smoothing + Confidence smoothing
        return torch.cat([
            (temp_j * alpha.unsqueeze(-1).expand_as(temp_j)),
            (conf_i - conf_j).unsqueeze(-1)], -1)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}{self.out_channels}, heads={self.heads}')

class GpNet(nn.Module):
    def __init__(self,  inchannel,num_nodes, num_class, heads, bias, device):
        super().__init__()
        print('inchannel',inchannel)
        self.model =  GCNConv(inchannel, 128)
        self.num_nodes = num_nodes
        self.cagat = ProbAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         num_nodes=num_nodes,
                                         heads=heads,
                                         bias=bias,
                                         device = device)
        self.device = device
        
    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits,edge_index)
        return temperature

    def graph_temperature_scale(self, logits,edge_index):

        temperature = self.cagat(logits,edge_index).view(self.num_nodes, -1)
        return temperature


class Pool(torch.nn.Module):
    def __init__(self, in_channels, ratio, Conv=ChebConv, non_linearity=torch.sigmoid):
        super(Pool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None, negative_score=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x, edge_index).squeeze()
        score = self.non_linearity(score)
        if negative_score is None:
            negative_score = torch.zeros(np.shape(score), dtype=torch.float).cuda()
        score = self.non_linearity(score - negative_score)

        perm = topk(score, self.ratio, batch)
        x = x[perm]
        batch = batch[perm]

        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score

class SelfAttnConv(MessagePassing):
    def __init__(self, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(SelfAttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        self.attn_lin = nn.Linear(attn_dim, 1)

    def forward(self, h, edge_index, edge_attr=None, h_attn=None, **kwargs):
       
        if edge_index is None:
            h_attn = h_attn if h_attn is not None else h
            attn_weights = self.attn_linear(h_attn).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.mm(attn_weights, h)
   
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self,edge_index, h_j, edge_attr, h_attn_j):

        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
  
        index = edge_index[0]
      
        a_j = self.attn_lin(h_attn)
        a_j = softmax(a_j, index)
        t = h_j * a_j
        return t

    def update(self, aggr_out):
        return aggr_out

class Sub_leafNet(nn.Module):
    def __init__(self, inchannel,num_nodes, in_size, hidden_size, out_size, dropout, device):
        super(Sub_leafNet, self).__init__()


        self.dropout = dropout
        self.linear = nn.Linear(num_nodes, in_size, bias=True)
        self.pe_feat = torch.FloatTensor(torch.eye(num_nodes)).to(device)
        self.pool = Pool(inchannel,0.5)
        self.attn = SelfAttnConv(inchannel)
        self.subgraph_reduction = nn.Linear(inchannel, 128)
        self.pnet = GpNet(inchannel,num_nodes, 128, 4, 1, device)
    def forward(self, x, edge_index):

        prob = self.pnet(x,edge_index)
        
        edge_index = edge_index.cpu()
        #print('----------------edge_index',edge_index)
        g = dgl.graph((edge_index[0],edge_index[1]))
        sub_list = [] 
        for i in range(g.num_nodes()):
            a = [i]
            traces, types = random_walk(g, nodes=i, length=5, prob=prob)
            sampled_nodes, _, _, _ = pack_traces(traces, types)
            b = np.unique(sampled_nodes.asnumpy())
            from itertools import combinations,product
            s_list = [list(i) for i in list(product(a,b))]
            sub_list.extend(s_list)
        
        new_subgraph = []
        subgraph_emb = []
        vector_subgraphs = []
        for sub_G in sub_list:
            if len(sub_G) !=0 :
                edge = torch.tensor(np.array(sub_G).T).cuda()
                a = self.pool(x,edge)
                if a[1].shape[1] != 0:
                    new_subgraph.append(a)      
                    b = self.attn(x,a[1])
                    vector_subgraph = self.subgraph_reduction(b)
                    subgraph_emb.append(b)
                    vector_subgraphs.append(vector_subgraph)

        for num in range(1,len(vector_subgraphs)):
            vector_subgraphs[0] += vector_subgraphs[num]
            aggregate_emb = vector_subgraphs[0]

            
        return aggregate_emb




class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Pl(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionG1(nn.Module):
    def __init__(self):
        super(AttentionG1, self).__init__()
        kernel_size = 7
        self.compress = Pl()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        return x 
class AttentionG2(nn.Module):
    def __init__(self):
        super(AttentionG2, self).__init__()
        kernel_size = 7
        self.compress = Pl()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale
class FuseAttention(nn.Module):
    def __init__(self):
        super(FuseAttention, self).__init__()
        self.g1 = AttentionG1()
        self.g2 = AttentionG2()
    def forward(self, x,x2,x3):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.g1(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()

        x2_perm2 = x2.permute(0,3,2,1).contiguous()
        x2_out2 = self.g1(x2_perm2)
        x2_out21 = x2_out2.permute(0,3,2,1).contiguous()

        x3_perm2 = x3.permute(0,3,2,1).contiguous()
        x3_out2 = self.g1(x3_perm2)
        x3_out21 = x3_out2.permute(0,3,2,1).contiguous()

        #x_out = torch.concat((x_out11 , x2_out21),axis=3)
        x_out = self.attention( x_out11 , x2_out21, x3_out21, mask=None, dropout=None)
        x_out_perm2 = x_out.permute(0,3,2,1).contiguous()
        x_out_out2 = self.g2(x_out_perm2)
        x_out_out21 = x_out_out2.permute(0,3,2,1).contiguous()

        
        x_out_out21  = torch.squeeze(x_out_out21,dim=0)
        x_out_out21  = torch.squeeze(x_out_out21,dim=0)

        return x_out_out21

    def attention(self, query, key, value, mask=None, dropout=None):
       
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'),)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)












class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhid_feat, nhid_stru, tau=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.tau = tau

        self.feat2stu = torch.nn.Linear(nhid_feat, nhid)
        self.stru2stu = torch.nn.Linear(nhid_stru, nhid)

    def forward(self, adj, x):
        #imp[0]
        imp = torch.zeros([x.shape[0], x.shape[1]]).cuda()
        # print(imp)
        # print(x)
        x = torch.where(torch.isnan(x), imp, x)

        middle_representations = []
        h = self.gc1(x, adj)
        middle_representations.append(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj)
        middle_representations.append(h)

        return h, middle_representations

    #contrast loss
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
             mean: bool = True):
       
        R_stu_1 = z1#[0]
        R_fea_1 = self.feat2stu(z2)
        R_str_1 = self.stru2stu(z3)
        fea_stu_1 = self.semi_loss(R_stu_1, R_fea_1)
        str_stu_1 = self.semi_loss(R_stu_1, R_str_1)
        fea_stu_1 = fea_stu_1.mean() if mean else fea_stu_1.sum()
        str_stu_1 = str_stu_1.mean() if mean else str_stu_1.sum()

        R_stu_2 = z1#[1]
        R_fea_2 = z2#[1]
        R_str_2 = z3#[1]
        fea_stu_2 = self.semi_loss(R_stu_2, R_fea_2)
        str_stu_2 = self.semi_loss(R_stu_2, R_str_2)
        fea_stu_2 = fea_stu_2.mean() if mean else fea_stu_2.sum()
        str_stu_2 = str_stu_2.mean() if mean else str_stu_2.sum()

        loss_mid_fea = fea_stu_1 + fea_stu_2
        loss_mid_str = str_stu_1 + str_stu_2

        return loss_mid_fea, loss_mid_str

