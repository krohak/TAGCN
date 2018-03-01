import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

adj = normalize_adj(adj) # normalize_adj(adj+sp.eye(adj.shape[0]))
adj = adj.toarray()

G = nx.Graph(adj)

def memoize(f):
    memo={}

    def helper(g,x,y,z):
        if (x,y,z) not in memo:
            if (y,x,z) not in memo:
                p=f(g,i=x,j=y,cutoff=z)
                memo[(x,y,z)]=p
                print(x,y,z,p)
                return memo[(x,y,z)]

            return memo[(y,x,z)]
        return memo[(x,y,z)]

    return helper


@memoize
def calculate_path_weight(G, i, j, cutoff):
    """https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.simple_paths.all_simple_paths.html"""
    # put G = nx.Graph(adj) before calling the function
    paths = nx.all_simple_paths(G, source=i, target=j,cutoff=cutoff)
    paths=[p for p in paths]
    return len(paths)


@memoize
def calculate_path_weight_norm(G, i, j, cutoff):
    paths = nx.all_simple_paths(G, source=i, target=j,cutoff=cutoff)

    path_weight = 0
    for path in paths:
        i = 1
        weight = 1
        while i < len(path):
            weight *= adj[path[i-1]][path[i]]
            i+=1
        path_weight += weight

    return path_weight



#calculate_path_weight(G,0,1378,2)

Kl=3 #for K=2

saved = np.zeros(shape=(features.shape[0],features.shape[0],2))

for k in range(1,Kl):
	for i in range(features.shape[0]):
		for j in range(features.shape[0]):
			saved[i][j][k-1]=calculate_path_weight_norm(G,i,j,k)


saved.dump("pubmed_path_weights_norm.dat")
