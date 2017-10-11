from __future__ import print_function
import numpy as np
import os
import sys
from pdb import set_trace
from mnist import MNIST

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_mnist(path):
    mndata = MNIST(path)
    mnist = mndata.load_training()
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
    final_episode = list()
    for (state, action, reward_lab, qval_lab) in episode:
        final_im = np.zeros((28, 28, len(state)))
        for (i, num) in enumerate(state):
            im_index = np.random.randint(len(label_to_im_dict[num]))
            im = label_to_im_dict[num][im_index].reshape((28, 28))
            final_im[:, :, i] = im
        final_episode.append(final_im, action) 

def main():
    mnist = load_mnist('data/') 
    mnist_label_to_image(mnist, 100)

if __name__=='__main__':
    main()

    
