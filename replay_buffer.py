import numpy as np
import random

class ReplayBuffer(object):


    def __init__(self, N):
        self.N = N
        self.lst = list()

    def store(self, state, act, reward, new_state, _mdp_new_states):
        if len(self.lst) == self.N:
            self.lst.pop(0)
        self.lst.append((state, act, reward, new_state, _mdp_new_states))

    def sample(self, num):
        if num > len(self.lst):
            return self.lst
        return random.sample(self.lst, num)

    def full(self):
        return len(self.lst) == self.N

    def get_buffer(self):
        return self.lst

    def get_size(self):
        return len(self.lst)
