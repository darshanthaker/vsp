import matplotlib.pyplot as plt
import numpy as np
import pickle
from util import *
from pdb import set_trace

def accuracy(baseline, nondet):
    plt.plot(baseline, label='Baseline', marker='s', \
             markersize=5, linewidth=2)
    plt.plot(nondet, label='Non-deterministic', marker='p', \
             markersize=5, linewidth=2, linestyle='dashed')
    plt.xlabel('Number of training epochs')
    plt.ylabel('Testing accuracy')
    plt.legend()
    plt.savefig('results1.png')
    #plt.show() 

def plot_action_freq(dir_name, epoch):
    freqs = dict()
    for i in range(1, 9):
        name = 'pred_action_freq_{}_{}'.format(epoch, i)
        with open('{}/{}'.format(dir_name, name), 'rb') as fp:
            freq = pickle.load(fp)
            freq = normalize_dict(freq, 100)
            lst = list()
            pos = 0
            for (k, v) in freq.items():
                if k[1] == 1:
                    pos += v
            lst.append(pos)
            lst.append(1.0 - pos)
            freqs[i] = lst

    pos_lst = [v[0] for (k, v) in freqs.items()]
    neg_lst = [v[1] for (k, v) in freqs.items()]
    ind = np.arange(len(pos_lst))
    width = 0.35

    p1 = plt.bar(ind, pos_lst, width, color='#d62728', align='center')
    p2 = plt.bar(ind, neg_lst, width, bottom=pos_lst, align='center')

    plt.ylabel('Percent')
    plt.xticks(ind, ['({}, {})'.format(x, x) for x in range(1, 9)])
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend((p1[0], p2[0]), ('Add 1', 'Subtract 1'))
    plt.show()

def main():
    epoch = 1
    deterministic = True
    num_episodes = 50000
    deterministic_prob = 0.5
    dir_name = "{}_{}_{}".format(deterministic, num_episodes, deterministic_prob)
    plot_action_freq(dir_name, epoch)
    
     

if __name__=="__main__":
    main()
