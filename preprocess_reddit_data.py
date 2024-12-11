'''
For the Reddit dataset, our task is to use (solely) the network information to predict the embeddings of nodes.

Use the network built from title info only.
'''

import os
import pickle
import torch
import torch_geometric as pyg
import scipy
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
import datetime

from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_scipy_sparse_matrix

d2 = 6

temporal_edges = pd.read_csv('data_raw/reddit_hyperlink/soc-redditHyperlinks-title.tsv', sep='\t')
# Collapse the temporal edges into a single edge
# LINK_SENTIMENT can be either 1 or -1. We will count how many times the edge appears with a positive sentiment and how many times with a negative sentiment, and use them as two seperate edge features.
# This allows directed edges.
edges = []
edge2index = {}
username2node = {}
print(datetime.datetime.now())
for i, row in tqdm.tqdm(temporal_edges.iterrows(), total=temporal_edges.shape[0], desc='Collapsing temporal edges'):
    source = row['SOURCE_SUBREDDIT']
    target = row['TARGET_SUBREDDIT']
    if source == target:
        continue
    if source not in username2node:
        username2node[source] = len(username2node)
    if target not in username2node:
        username2node[target] = len(username2node)
    source = username2node[source]
    target = username2node[target]
    if (source, target) not in edge2index:
        edge2index[(source, target)] = len(edges)
        edges.append([source, target, 0, 0])
    if row['LINK_SENTIMENT'] == 1:
        edges[edge2index[(source, target)]][2] += 1
    else:
        edges[edge2index[(source, target)]][3] += 1
edges = pd.DataFrame(edges, columns=['source', 'target', 'positive', 'negative'])
print('Done collapsing temporal edges')
print('Number of edges:', edges.shape[0])

# Create a networkx directed graph from the edges fast
graph = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr=True, create_using=nx.DiGraph)

print('Reading embeddings...')
print(datetime.datetime.now())
embeddings = {}
with open('data_raw/reddit_hyperlink/web-redditEmbeddings-subreddits.csv', 'r') as f:
    for line in f:
        line = line.strip().split(',')
        if line[0] in username2node:
            embeddings[username2node[line[0]]] = np.array(list(map(float, line[1:])))
# Convert embeddings to a tensor (with missing values as NA)
nans = np.empty(300)
nans[:] = np.nan
embeddings = np.array([embeddings.get(node, nans) for node in graph.nodes])
print('Done reading embeddings')

print('Reading node info & shuffling & splitting datasets...')
print(datetime.datetime.now())
# Convert to a PyG Data object
graph = from_networkx(graph)
# Add embeddings as node-level target of prediction
graph.y = torch.tensor(embeddings, dtype=torch.float32)
# No node feature, so add a dummy one
graph.x = torch.ones(graph.num_nodes, 1, dtype=torch.float32)
# Add edge attributes
graph.edge_attr = torch.tensor(edges[['positive', 'negative']].values, dtype=torch.float32)

# Shuffle the node indices
perm = torch.randperm(graph.num_nodes)

# Split data into train, validation, and test sets
data = graph
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[perm[:int(0.6 * len(perm))]] = 1
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask[perm[int(0.6 * len(perm)):int(0.8 * len(perm))]] = 1
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[perm[int(0.8 * len(perm)):]] = 1

# Compute alter_edges and edge_attr
# `alter_edges` is a sparse matrix of shape (num_edges, num_nodes), each row being a one-hot encoding of the edges among a node's alters (a.k.a. alter-edge)
# `edge_attr` is a sparse matrix of shape (model.d2, num_edges), each column stacking 3 edge features: the alter-edge, the edge between the ego and one alter, the edge between the ego and the other alter
# alter_edges = scipy.sparse.dok_matrix((data.num_edges, data.num_nodes))
# edge_attr = torch.zeros(d2, data.num_edges)
# row_ind, col_ind, edge_ind = data.csr()
# # Enumerate over the nodes and look for alter-edges
# # Note that here the graph is directed. Therefore, j and k need not maintain order.
# print('Computing alter-edges...')
# print(datetime.datetime.now())
# for i in tqdm.trange(len(row_ind) - 1):
#     for j in range(row_ind[i], row_ind[i + 1]):
#         for k in range(row_ind[i], row_ind[i + 1]):
#             if j == k:
#                 continue
#             # Check if node col_index[j] and node col_index[k] are connected by another edge. If yes, it is an alter-edge. In other words, check for triangles.
#             alter1 = col_ind[j].item()
#             alter2 = col_ind[k].item()
#             if (alter1, alter2) in edge2index:
#                 alter_edges[edge2index[(alter1, alter2)], i] = 1
#                 edge_attr[:, edge2index[(alter1, alter2)]] = torch.cat([
#                     data.edge_attr[edge2index[(alter1, alter2)]],
#                     data.edge_attr[edge2index[(i, alter1)]],
#                     data.edge_attr[edge2index[(i, alter2)]]
#                 ])
# # Transform alter_edge from scipy.sparse.dok_matrix to torch.sparse.FloatTensor
# indices = torch.tensor(list(alter_edges.keys()), dtype=torch.long).t()
# values = torch.tensor(list(alter_edges.values()), dtype=torch.float)
# alter_edges = torch.sparse_coo_tensor(
#     indices,
#     values,
#     torch.Size([data.num_edges, data.num_nodes])
# )

alter_edges = [] # (ego-alter_edge pairs), each alter_edge is the index in edge_attr
edge_attr = [] # (d2, number of node-alter_edge pairs)
row_ind, col_ind, edge_ind = data.csr()

print('Computing alter-edges...')
print(datetime.datetime.now())
for i in tqdm.trange(len(row_ind) - 1):
    for j in range(row_ind[i], row_ind[i + 1]):
        for k in range(row_ind[i], row_ind[i + 1]):
            if j == k:
                continue
            # Check if node col_index[j] and node col_index[k] are connected by another edge. If yes, it is an alter-edge. In other words, check for triangles.
            alter1 = col_ind[j].item()
            alter2 = col_ind[k].item()
            if (alter1, alter2) in edge2index:
                alter_edges.append((len(edge_attr), i))
                edge_attr.append(torch.cat([
                    data.edge_attr[edge2index[(alter1, alter2)]],
                    data.edge_attr[edge2index[(i, alter1)]],
                    data.edge_attr[edge2index[(i, alter2)]]
                ]))

alter_edges = torch.sparse_coo_tensor(
    torch.tensor(list(zip(*alter_edges)), dtype=torch.long),
    torch.ones(len(alter_edges), dtype=torch.float),
    torch.Size([len(alter_edges), data.num_nodes])
)
edge_attr = torch.stack(edge_attr, dim=1)
print('Done computing alter-edges')
print('Number of node-alter_edge pairs:', edge_attr.shape[1])

print(datetime.datetime.now())

# Save the preprocessed data
with open('data/preprocessed_reddit_title_data.pkl', 'wb') as f:
    pickle.dump((data, alter_edges, edge_attr, username2node), f)

