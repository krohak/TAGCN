import tensorflow as tf
import numpy as np

from utils import *
from metrics import *
from inits import *

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


features = features.todense()

# g = tf.get_default_graph()
# [op.name for op in g.get_operations()]

path_weight_matrix = np.load("path_weights.dat")
path_weight_matrix = path_weight_matrix.astype('float32')


from inits import *
# var_gs = {}
name = '01'
Fl = 8
Kl = 2
Cl = features.shape[1]
Nl = features.shape[0]
from tensorflow.python.framework.ops import reset_default_graph
reset_default_graph()


features_m = tf.placeholder(tf.float32, shape=[features.shape[0], features.shape[1]])
pwm = tf.placeholder(tf.float32, [Nl, Nl, Kl])
outputs=[]

conv = np.zeros([Nl,Fl],dtype=np.float32)
#tf.placeholder(tf.float32, shape=[Nl,Fl])

# droupout
features_m = tf.nn.dropout(features_m, 1-FLAGS.dropout)

# layer 1
with tf.variable_scope( name + '_vars'):

    for k in range(Kl):

        w_k = pwm[:,:,k]

        s = tf.matmul(w_k,features_m)

        G_k = tf.get_variable(name=('G_k_%s'%(k)),initializer=tf.contrib.layers.xavier_initializer(),shape=[Cl,Fl])

        res = tf.matmul(s,G_k)

        outputs.append(res)

        conv = tf.add(conv,res)

        print(conv.shape)

        outputs.append(conv)

    # add bias
    initial = tf.zeros([Nl,Fl], dtype=tf.float32)
    bias = tf.Variable(initial, name='bias')

    conv = tf.add(conv,bias)

    # apply non-linearity
    conv = tf.nn.relu(conv)

# droupout
conv = tf.nn.dropout(conv, 1-FLAGS.dropout)

# layer 2
with tf.variable_scope( name + '_vars'):

    for k in range(Kl):

        w_k = pwm[:,:,k]

        s = tf.matmul(w_k,conv)

        G_k = tf.get_variable(name=('G_k2_%s'%(k)),initializer=tf.contrib.layers.xavier_initializer(),shape=[Fl,Fl])

        res = tf.matmul(s,G_k)

        outputs.append(res)

        conv = tf.add(conv,res)

        print(conv.shape)

        outputs.append(conv)

    # add bias
    initial = tf.zeros([Nl,Fl], dtype=tf.float32)
    bias = tf.Variable(initial, name='bias2')

    conv = tf.add(conv,bias)

    # apply non-linearity
    conv = tf.nn.relu(conv)

# droupout
conv = tf.nn.dropout(conv, 1-FLAGS.dropout)

# output
with tf.variable_scope( name + '_vars'):

    w_k = pwm[:,:,k]

    s = tf.matmul(w_k,conv)

    G_k = tf.get_variable(name=('G_k3_%s'%(k)),initializer=tf.contrib.layers.xavier_initializer(),shape=[Fl,7])

    conv = tf.matmul(s,G_k)


    print(conv.shape)
    # add bias
    initial = tf.zeros([Nl,7], dtype=tf.float32)
    bias = tf.Variable(initial, name='bias3')
    conv = tf.add(conv,bias)


    # apply non-linearity
    conv = tf.nn.relu(conv)


# for training
accuracy1 = masked_accuracy(conv,y_train, train_mask)
loss1= 0
loss1 += masked_softmax_cross_entropy(conv, y_train, train_mask)

optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
opt_op = optimizer.minimize(loss1)

# for testing
accuracy2 = masked_accuracy(conv,y_test, test_mask)
loss2= 0
loss2 += masked_softmax_cross_entropy(conv, y_test, test_mask)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(FLAGS.epochs):

    evals = sess.run([opt_op,loss1,accuracy1],feed_dict={features_m:features,pwm:path_weight_matrix})
    print(evals[1], evals[2])

outs_val = sess.run([loss2, accuracy2], feed_dict={features_m:features,pwm:path_weight_matrix})
print(outs_val[0], outs_val[1])
