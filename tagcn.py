import tensorflow as tf
import numpy as np

from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

path_weight_matrix = np.load("path_weights.dat")
path_weight_matrix.shape
path_weight_matrix = path_weight_matrix.astype('float32')

path_weight_matrix[1,:,1].shape

f=features.todense()
output=[]
output = np.asarray(output,dtype=np.float32)
f.shape
f.dtype, path_weight_matrix.dtype

from inits import *
var_gs = {}
name = '01'
Fl = 8
Kl = 2
Cl = features.shape[1]
Nl = features.shape[0]
from tensorflow.python.framework.ops import reset_default_graph
reset_default_graph()

f = tf.placeholder(tf.float32, [Nl, Cl])
path_weight_matrix = tf.placeholder(tf.float32, [Nl, Nl, 2])
output = tf.placeholder(tf.float32, [None])


with tf.variable_scope( name + '_vars'):
    for i in range(features.shape[0]):
        conv = tf.get_variable("conv", shape=[1,Fl])
        for k in range(Kl):
            for c in range(Cl):
                    var_gs = tf.get_variable(name='var_gs',initializer=tf.contrib.layers.xavier_initializer(),shape=[1,Fl])
                    w_k = path_weight_matrix[i,:,k]
                    x_c = f[0:,c]
                    s = tf.matmul(tf.transpose(tf.expand_dims(w_k, 1)),tf.expand_dims(x_c, 1))
                    conv = tf.add(conv,tf.multiply(s[0,0],var_gs))
		    tf.get_variable_scope().reuse_variables()
        print(conv)

