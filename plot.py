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

def plot_lossesRLVSIL(dir_names):
    plt.clf()
    with open(dir_names[0] + '/RL_losses', 'rb') as fp:
        RL5 = pickle.load(fp)
        RL5 = RL5[:50]
    with open(dir_names[1] + '/losses', 'rb') as fp:
        IL = pickle.load(fp)
        IL = IL[:50]
    with open(dir_names[2] + '/losses', 'rb') as fp:
        IL1 = pickle.load(fp)
        IL1 = IL1[:30]
    with open(dir_names[2] + '/RL_losses', 'rb') as fp:
        RLft = pickle.load(fp)
        RLft = RLft[:20]
        ILRL = IL1 + RLft
    plt.plot(RL5, label='RL', marker='s', \
             markersize=5, linewidth=2)
    plt.plot(IL, label='IL', marker='p', \
             markersize=5, linewidth=2)
    plt.plot(ILRL, label='IL + RL', marker='p', \
             markersize=5, linewidth=2)
    plt.xlabel('Number of training epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('RLVSIL.png')

def plot_losses(dir_names):
    plt.clf()
    with open(dir_names[0] + '/losses', 'rb') as fp:
        det = pickle.load(fp)
        det = det[:50]
    with open(dir_names[1] + '/losses', 'rb') as fp:
        nondet5 = pickle.load(fp)
        nondet5 = nondet5[:50]
    with open(dir_names[2] + '/losses', 'rb') as fp:
        nondet6 = pickle.load(fp)
        nondet6 = nondet6[:50]
    with open(dir_names[3] + '/losses', 'rb') as fp:
        nondet7 = pickle.load(fp)
        nondet7 = nondet7[:50]
    with open(dir_names[4] + '/losses', 'rb') as fp:
        nondet8 = pickle.load(fp)
        nondet8 = nondet8[:50]
    with open(dir_names[5] + '/losses', 'rb') as fp:
        nondet9 = pickle.load(fp)
        nondet9 = nondet9[:50]
    plt.plot(det, label='Prob 1', marker='s', \
             markersize=5, linewidth=2)
    plt.plot(nondet5, label='Prob 0.5', marker='p', \
             markersize=5, linewidth=2)
    plt.plot(nondet6, label='Prob 0.6', marker='p', \
             markersize=5, linewidth=2)
    plt.plot(nondet7, label='Prob 0.7', marker='p', \
             markersize=5, linewidth=2)
    plt.plot(nondet8, label='Prob 0.8', marker='p', \
             markersize=5, linewidth=2)
    plt.plot(nondet9, label='Prob 0.9', marker='p', \
             markersize=5, linewidth=2)
    plt.xlabel('Number of training epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('nondet_losses.png')

def plot_action_freq(dir_name, epoch, RL=False):
    freqs = dict()
    for i in range(1, 9):
        if RL:
            name = 'RL_pred_action_freq_{}_{}'.format(epoch, i)
        else:
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
    plt.savefig('RL_IL_epoch{}'.format(epoch))
    plt.show()

def main():
    dir_names = ["RL5", "True_10000_0.5", "RL_IL"]
    #plot_lossesRLVSIL(dir_names)

    dir_names = ["True_10000_0.5", "False_10000_0.5", "False_10000_0.6", \
                 "False_10000_0.7", "False_10000_0.8", "False_10000_0.9"]
    #plot_losses(dir_names)

    epoch = 110 
    dir_name = "RL_IL"
    plot_action_freq(dir_name, epoch, RL=True)
     

if __name__=="__main__":
    main()
