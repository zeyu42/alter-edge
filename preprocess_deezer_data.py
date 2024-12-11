'''
For the Deezer dataset, our task is to use the network information to predict the transitivity (a.k.a. local clustering coefficient) of nodes. For this purpose, we discard all the nodal features present in the dataset.
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

d2 = 3

edges = pd.read_csv('data_raw/deezer_europe/deezer_europe_edges.csv')
edge2index = {}
node2index = {}
for i, row in edges.iterrows():
    if row['node_1'] not in node2index:
        node2index[row['node_1']] = len(node2index)
    if row['node_2'] not in node2index:
        node2index[row['node_2']] = len(node2index)
    edge2index[(node2index[row['node_1']], node2index[row['node_2']])] = i
    edge2index[(node2index[row['node_2']], node2index[row['node_1']])] = i
graph = nx.from_pandas_edgelist(edges, source='node_1', target='node_2')

# Compute the transitivity of each node
transitivity = nx.clustering(graph, nodes=graph.nodes)
transitivity = [[transitivity[node]] for node in graph.nodes]

# Convert to a PyG Data object
graph = from_networkx(graph)
# Add transitivity as node-level target of prediction
graph.y = torch.tensor(transitivity, dtype=torch.float32)
# No node feature, so add a dummy one
graph.x = torch.ones(graph.num_nodes, 1, dtype=torch.float32)
# No edge attribute, so add a dummy one (LCC is essentially counting)
graph.edge_attr = torch.ones(graph.num_edges, 1, dtype=torch.float32).reshape(-1, 1)

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
# print('Computing alter-edges...')
# print(datetime.datetime.now())
# for i in tqdm.trange(len(row_ind) - 1):
#     for j in range(row_ind[i], row_ind[i + 1] - 1):
#         for k in range(j + 1, row_ind[i + 1]):
#             # Check if node col_index[j] and node col_index[k] are connected by another edge. If yes, it is an alter-edge. In other words, check for triangles.
#             alter1 = col_ind[j]
#             alter2 = col_ind[k]
#             # Here, alter1 < alter2
#             for l in range(row_ind[alter1], row_ind[alter1 + 1]):
#                 if col_ind[l] == alter2:
#                     alter_edges[edge_ind[col_ind[l]], i] = 1
#                     edge_attr[:, edge_ind[col_ind[l]]] = torch.cat([
#                         data.edge_attr[edge_ind[col_ind[l]]],
#                         data.edge_attr[edge_ind[j]],
#                         data.edge_attr[edge_ind[k]]
#                     ])
#                     break
# # Transform alter_edge from scipy.sparse.dok_matrix to torch.sparse.FloatTensor
# alter_edges = torch.sparse_coo_tensor(
#     torch.tensor(list(alter_edges.keys())).t(),
#     torch.tensor(list(alter_edges.values())).float(),
#     torch.Size([data.num_edges, data.num_nodes])
# )
alter_edges = [] # (alter_edge-ego pairs), each alter_edge is the index in edge_attr
edge_attr = [] # (d2, number of node-alter_edge pairs)
row_ind, col_ind, edge_ind = data.csr()

print('Computing alter-edges...')
print(datetime.datetime.now())
for i in tqdm.trange(len(row_ind) - 1):
    for j in range(row_ind[i], row_ind[i + 1] - 1):
        for k in range(j + 1, row_ind[i + 1]):
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
with open('data/preprocessed_deezer_data.pkl', 'wb') as f:
    pickle.dump((data, alter_edges, edge_attr), f)

