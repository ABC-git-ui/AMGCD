import os
import random
import torch

import scipy.sparse as sp
import numpy as np
from tensorflow.keras.utils import to_categorical
from typing import Optional
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import csr_matrix,lil_matrix
import torch
from collections import Counter
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import precision_recall_curve,roc_curve,auc,balanced_accuracy_score
from scipy import interp
import math
from sklearn.model_selection import KFold,StratifiedKFold
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, to_undirected, to_networkx
import networkx as nx


from itertools import chain
def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask
def apply_feature_mask(features, mask):
    features[mask] = float('nan')


from sklearn.utils import shuffle
def load_data(dataset, repeat, device):
    path = './dataset/{}/'.format(dataset)

    f = pd.read_csv(path + '{}_feature.csv'.format(dataset), header=None).values
    adj = pd.read_csv(path+ '{}_AssociationMatrix.csv'.format(dataset),header=None)

    edge = reconstruct(adj)
    edge_true = edge.loc[edge[2]==1]
    edge_false = edge.loc[edge[2]==0]
    edge_false = edge_false.iloc[:len(edge_true),]
    edge = pd.concat([edge_true,edge_false],ignore_index=True)
    edge = shuffle(edge)
 
    l = edge[2]
 
    #kfold = StratifiedKFold(n_splits=5,shuffle=False,random_state=None)
    kfold = KFold(n_splits=5,shuffle=False,random_state=None)
    trs = []
    tes = []
    for tr, te in kfold.split(edge[[0,1]]):
        trs.append(tr)
        tes.append(te)

    test = tes[repeat]
    train = trs[repeat]
    val = tes[repeat]

    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
 
    idx_test = test#.tolist()
    idx_train = train#.tolist()
    idx_val = val#.tolist()

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)

    label = torch.LongTensor(np.array(edge.iloc[train][2])).to(device)
   
    label_oneHot = torch.FloatTensor(to_categorical(edge.iloc[train][2])).to(device)

    label_test = torch.LongTensor(np.array(edge.iloc[test][2])).to(device)

    label_oneHot_test = torch.FloatTensor(to_categorical(edge.iloc[test][2])).to(device)
    edge_train_all = edge.iloc[train]
    #edge_train = edge.loc[edge[2]==1]
    edge_train = edge_train_all.loc[edge_train_all[2]==1]
 
    edge_index_train = torch.from_numpy(np.array(list(zip(edge_train[0],edge_train[1])))).T
   # print(edge_index_train.shape)
    # a,b = np.where(edge_train.values!=0) 
    # edge_index_train = torch.from_numpy(np.vstack([a,b]))
    # edge_index_tree, _, _, _, _ =tree_decomposition(edge_index_train,features.shape[0])
    # edge_index_tree = pd.DataFrame(edge_index_tree.T.numpy())
    #print('edge_index_tree',edge_index_tree)
    #print('features',features)
    sadj = sp.coo_matrix((np.ones(edge_train.shape[0]), (edge_train.iloc[:, 0].tolist(), edge_train.iloc[:, 1].tolist())),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    #sadj = edge_delete(rate, sadj)

    ttadj = sadj + sp.eye(sadj.shape[0])
    ttadj = torch.FloatTensor(ttadj.todense()).to(device)
    # A
    tadj = torch.FloatTensor(sadj.todense()).to(device)
    # stu_input
    sadj = normalize_sparse(sadj + sp.eye(sadj.shape[0]))
    nsadj = torch.FloatTensor(np.array(sadj.todense())).to(device)

 
    edge_test_all = edge.iloc[test]
    edge_test = edge_test_all.loc[edge_test_all[2]==1]
    #edge_test = edge.loc[edge[2]==1]
    edge_index_test = torch.from_numpy(np.array(list(zip(edge_test[0],edge_test[1])))).T
    # a,b = np.where(edge_test.values!=0) 
    # edge_index_test = torch.from_numpy(np.vstack([a,b]))
    # edge_index_tree, _, _, _, _ =tree_decomposition(edge_index_test,features.shape[0])
    # edge_index_tree_test = pd.DataFrame(edge_index_tree.T.numpy())

    sadj_test = sp.coo_matrix((np.ones(edge_test.shape[0]), (edge_test.iloc[:, 0].tolist(), edge_test.iloc[:, 1].tolist())),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj_test = sadj_test + sadj_test.T.multiply(sadj_test.T > sadj_test) - sadj_test.multiply(sadj_test.T > sadj_test)
    #sadj_test = edge_delete(rate, sadj_test)

    # ppr_input:A+I
    ttadj_test = sadj_test + sp.eye(sadj_test.shape[0])
    ttadj_test = torch.FloatTensor(ttadj_test.todense()).to(device)
    # A
    tadj_test = torch.FloatTensor(sadj_test.todense()).to(device)
    # stu_input
    sadj_test = normalize_sparse(sadj_test + sp.eye(sadj_test.shape[0]))
    nsadj_test = torch.FloatTensor(np.array(sadj_test.todense())).to(device)

    return  tadj, nsadj, tadj_test, nsadj_test, features, label, edge_index_train, edge_train_all, edge_test_all,label_test,edge_index_test

    #return ttadj, tadj, nsadj,ttadj_test, tadj_test, nsadj_test, features, label, edge_index_train, idx_train, idx_val, idx_test, edge_train_all, edge_test_all,label_test,edge_index_test






def reconstruct(pi_dis):
    score=[]
    pi=[]
    dis=[]
    for i in range(len(pi_dis)):
        for j in range(pi_dis.shape[1]):
            score.append(pi_dis.iloc[i,j])
            pi.append(i)
            dis.append(j+len(pi_dis))
    return pd.DataFrame({0:pi,1:dis,2:score})
def partition(ls, size):
    # """
    # Returns a new list with elements
    # of which is a list of certain size.
    #
    #     >>> partition([1, 2, 3, 4], 3)
    #     [[1, 2, 3], [4]]
    #     【1，2，3，4，5，6】，3 ==》 【1 2 3】【4 5 6】
    # """
    return [ls[i:i+size] for i in range(0, len(ls), size)]


def auroc(prob,label):
    y_true=label.data.cpu().numpy().flatten()
    y_scores=prob.data.cpu().numpy().flatten()
    fpr,tpr,thresholds=roc_curve(y_true,y_scores)
    auroc_score=auc(fpr,tpr)
    return auroc_score,fpr,tpr

def auprc(prob,label):
    y_true=label.data.cpu().numpy().flatten()
    y_scores=prob.data.cpu().numpy().flatten()
    precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
    auprc_score=auc(recall,precision)
    return auprc_score,precision,recall




from  collections import Iterable 
def flatten(items,ignore_types=(str,bytes)):
    for x in items:
        if isinstance(x,Iterable) and not isinstance(x,ignore_types):
            yield from flatten(x)
        else:
            yield x

def prediction(predlabel,labels):
    predlabel_s=[]
    labels_s=[]
    for x in flatten(predlabel):
        predlabel_s.append(x)
    for x in flatten(labels):
        labels_s.append(x)

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    acc=accuracy_score(labels_s, predlabel_s)
    precision=precision_score(labels_s, predlabel_s)
    recall=recall_score(labels_s, predlabel_s)
    f1=f1_score(labels_s, predlabel_s)
    return acc,precision,recall,f1



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
              np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def sparse_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edges_s = sparse_mx.nonzero()
    edge_index_s = torch.tensor(np.vstack((edges_s[0], edges_s[1])), dtype=torch.long)
    return edge_index_s





def dense_to_sparse_tensor(matrix):
    rows, columns = torch.where(matrix > 0)
    values = torch.ones(rows.shape)
    indices = torch.from_numpy(np.vstack((rows,
                                          columns))).long()
    shape = torch.Size(matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)





