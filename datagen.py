import numpy as np
from mdp import *
from util import *

class DataGenerator(object):


    def __init__(self, mdp):
        self.mdp = mdp
        self.initial_state = self.mdp.get_initial_state()
        self.gamma = 0.99
        self.values = self.value_iteration(100)
        self.policy = self.find_optimal_policy()
        mnist = load_mnist('data/')
        self.label_to_im_dict = mnist_label_to_image(mnist)

    def find_optimal_policy(self):
        gamma = self.gamma
        policy = {s: (0, 0) for s in self.values.keys()}
        for state in self.values.keys():
            actions = self.mdp.get_all_actions(state)
            max_val = -float('inf')
            optimal_act = (0, 0)
            for a in actions:
                successors = self.mdp.get_transition_probs(state, a)        
                tmp_val = 0
                for (s, prob) in successors:
                    s = tuple(s)
                    reward = self.mdp.get_reward(state, a, s)
                    tmp_val += prob * (reward + gamma*self.values[s])
                if tmp_val > max_val:
                    max_val = tmp_val
                    optimal_act = a
            policy[state] = optimal_act
        return policy

    def value_iteration(self, num_iterations, gamma=1):
        gamma = self.gamma
        all_states = self.mdp.get_all_states()
        values = {tuple(s): 0 for s in all_states}
        values[self.mdp.goal_state] = 0
        for i in range(num_iterations):
            for state in all_states:
                state = tuple(state)
                actions = self.mdp.get_all_actions(state)
                max_val = -float('inf')
                for a in actions:
                    successors = self.mdp.get_transition_probs(state, a)
                    tmp_val = 0
                    for (s, prob) in successors: 
                        s = tuple(s)
                        reward = self.mdp.get_reward(state, a, s)
                        tmp_val += prob * (reward + gamma*values[s])
                    if tmp_val > max_val:
                        max_val = tmp_val
                values[state] = max_val
        eprint("[debug] finished value iteration")
        return values

    def gen_episodes(self, num_episodes, path):
        episodes = list()
        self.mdp.reset()
        for i in range(num_episodes):
            if i % 1000 == 0 and i != 0:
                eprint("{} episodes generated".format(i))
            episode = self.gen_episode()
            episodes.extend(episode)
        eprint("[debug] finished generating episodes")
        episodes = np.array(episodes)
        with open(path, 'wb') as f:
            np.save(f, episodes)
        return episodes
    
    def gen_episode(self, eps=0.2):
        episode = list()
        self.mdp.reset()
        while not self.mdp.at_goal():
            greedy_prob = np.random.random()
            if greedy_prob >= eps:
                episode.append([self.mdp.current_state, \
                    self.policy[self.mdp.current_state]])
                self.mdp.do_transition(self.mdp.current_state, \
                    self.policy[self.mdp.current_state])
            else:
                possible_actions = self.mdp.get_all_actions(self.mdp.current_state)
                action_prob = np.random.random() * len(possible_actions) 
                action_index = int(np.floor(action_prob))
                episode.append([self.mdp.current_state, possible_actions[action_index]]) 
                self.mdp.do_transition(self.mdp.current_state, \
                    possible_actions[action_index])
        episode = self.add_labels(episode)
        episode = state_to_image(episode, self.label_to_im_dict)
        return episode

    def action_to_index(self, action):
        all_actions = self.mdp.get_all_actions()
        return all_actions.index(action)

    def get_estimated_qval(self, state, action, truncated_episode):
        assert truncated_episode[0][0] == state and truncated_episode[0][1] == action
        qval = 0
        for (i, (state, action)) in enumerate(truncated_episode):
            reward = self.mdp.get_reward(state, action)
            qval += self.gamma**i * reward
        return qval

    def add_labels(self, episode):
        for (i, ep_lst) in enumerate(episode):
            state = ep_lst[0]
            action = ep_lst[1]
            reward_lab = self.mdp.get_reward(state, action)
            qval_lab = self.get_estimated_qval(state, action, episode[i:])
            ep_lst[1] = self.action_to_index(action)
            ep_lst.append(reward_lab)
            ep_lst.append(qval_lab)
        return episode
