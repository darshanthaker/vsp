from __future__ import print_function
import numpy as np
import os
import sys
from pdb import set_trace
from mnist import MNIST
import tensorflow as tf

NUM_DIGITS = 2
NUM_EPISODES = 10000
DETERMINISTIC_PROB = 0.5

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def mkdir(path):
    os.system("mkdir -p {}".format(path))

def load_mnist(path):
    mndata = MNIST(path)
    mnist = mndata.load_training()
    return mnist

def load_mnist_test(path):
    mndata = MNIST(path)
    mnist = mndata.load_testing()
    return mnist

def mnist_label_to_image(mnist):
    images = np.array(mnist[0])
    labels = np.array(mnist[1])
    assert images.shape[0] == labels.shape[0]
    y_to_x = dict()
    for i in range(images.shape[0]):
        label = labels[i]
        im = images[i][:]
        if label not in y_to_x:
            y_to_x[label] = [im]
        else:
            y_to_x[label].append(im)
    eprint("[debug] parsed mnist") 
    return y_to_x

def state_to_image(episode, label_to_im_dict):
    for (j, (state, action, reward_lab, qval_lab)) in enumerate(episode):
        final_im = np.zeros((28, 28, len(state)))
        for (i, num) in enumerate(state):
            im_index = np.random.randint(len(label_to_im_dict[num]))
            im = label_to_im_dict[num][im_index].reshape((28, 28))
            final_im[:, :, i] = im
        episode[j][0] = final_im
    return episode

def conv_layer(input, num_outputs, kernel_size, stride=1):
    return tf.contrib.layers.conv2d(input, num_outputs=num_outputs,
        kernel_size=kernel_size, stride=stride, weights_regularizer=tf.nn.l2_loss)

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  
