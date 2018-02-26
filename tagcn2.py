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

pwm = np.load("path_weights.dat")
pwm = pwm.astype('float32')

feat=features.todense()

from inits import *
var_gs = {}
name = '01'
Fl = 8
Kl = 2
#Cl = features.shape[1]
#Nl = features.shape[0]
Cl = 10
Nl = 10
from tensorflow.python.framework.ops import reset_default_graph
reset_default_graph()

f = tf.placeholder(tf.float32, [Nl, Cl])
path_weight_matrix = tf.placeholder(tf.float32, [Nl, Nl, 2])
#output = tf.placeholder(tf.float32, [None])


#f = tf.get_variable(name='f',dtype=tf.float32, shape=[Nl, Cl])
#path_weight_matrix = tf.get_variable(name='path_weight_matrix',dtype=tf.float32, shape=[Nl, Nl, 2])
#output = tf.get_variable(name='output',dtype=tf.float32, shape=[1,Fl])
outputs=[]

with tf.variable_scope( name + '_vars'):
    for i in range(Nl):
        conv = tf.get_variable("conv", shape=[1,Fl])
        for k in range(Kl):
            for c in range(Cl):
                    var_gs = tf.get_variable(name='var_gs',initializer=tf.contrib.layers.xavier_initializer(),shape=[1,Fl])
                    w_k = path_weight_matrix[i,:,k]
                    x_c = f[0:,c]
                    s = tf.matmul(tf.transpose(tf.expand_dims(w_k, 1)),tf.expand_dims(x_c, 1))
                    conv = tf.add(conv,tf.multiply(s[0,0],var_gs))
		    tf.get_variable_scope().reuse_variables()
		    print(conv.shape)
	outputs.append(conv)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(outputs, feed_dict={path_weight_matrix: pwm[:10,:10],f: feat[:10,:10]}))
