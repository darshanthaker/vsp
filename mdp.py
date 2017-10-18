from util import *
import numpy as np

class MNISTMDP(object):


    def __init__(self, num_digits, deterministic=True):
        self.num_digits = num_digits
        self.deterministic = deterministic
        eprint("deterministic is {}".format(self.deterministic))
        #self.initial_state = tuple([5 for i in range(self.num_digits)])
        self.initial_state = tuple([np.random.randint(10) for i in range(self.num_digits)])
        self.terminal_state = tuple([9 for i in range(self.num_digits)])
        self.terminal_state2 = tuple([0 for i in range(self.num_digits)])
        self.goal_state = tuple(['GOAL'])
        self.current_state = self.initial_state

        self.invalid_reward = -5
        self.goal_reward = 10

    def get_random_initial_state(self):
        return tuple([np.random.randint(10) for i in range(self.num_digits)])

    def reset(self):
        self.initial_state = tuple([np.random.randint(10) for i in range(self.num_digits)])
        self.current_state = self.initial_state

    def _take_deterministic_action(self, state, action):
        if action == (-1, -1):
            return self.goal_state
        new_state = list(state)
        if (new_state[action[0]] + action[1]) < 0:
            return tuple(new_state)
        new_state[action[0]] = (new_state[action[0]] + action[1]) % 10
        return tuple(new_state)

    def get_all_states(self):
        all_digits = range(int('9' * self.num_digits) + 1)
        return [[int(i) for i in str(x).zfill(self.num_digits)] for x in all_digits]

    def get_initial_state(self):
        return self.initial_state

    def get_current_state(self):
        return self.current_state

    def get_all_actions(self, state=None):
        if state is None:
            return [(i, 1) for i in range(self.num_digits)] + \
                   [(i, -1) for i in range(self.num_digits)] + \
                   [(-1, -1)]
        else:
            if self.at_terminal(state=state):
                return [(-1, -1)]
            return [(i, 1) for i in range(self.num_digits)] + \
                   [(i, -1) for i in range(self.num_digits)]

    def get_num_actions(self):
        return len(self.get_all_actions())

    def get_transition_probs(self, state, action):
        successors = list()
        if self.at_goal(state=state):
            return [(self.goal_state, 1)]
        if self.deterministic:
            new_state = self._take_deterministic_action(state, action)
            successors.append((new_state, 1))
            for act in self.get_all_actions(state):
                if act == action:
                    continue
                new_state = self._take_deterministic_action(state, act)
                successors.append((new_state, 0))
        else:
            # Assume uniform probability over all possible actions != action
            # and 0.5 for action
            num_actions = len(self.get_all_actions(state))
            new_state = self._take_deterministic_action(state, action)
            if num_actions == 1: # Only if terminal state, so edge case.
                successors.append((new_state, 1))
                return successors
            else:
                successors.append((new_state, DETERMINISTIC_PROB))
            uniform_prob = (1.0 - DETERMINISTIC_PROB)/(num_actions - 1)
            for act in self.get_all_actions(state):
                if act == action:
                    continue
                new_state = self._take_deterministic_action(state, act)
                successors.append((new_state, uniform_prob))
            sum_probs = sum([su[1] for su in successors])
            assert np.isclose(np.array([sum_probs]), np.array([1]))
        return successors
              
    def get_reward(self, state, action, new_state=None):
        if self.at_terminal(state=state):
            return self.goal_reward 
        if self.at_goal(state=state):
            return 0

        successors = self.get_transition_probs(state, action)
        if new_state is None:
            reward = 0
            for (s, prob) in successors:
                reward += prob * self.get_reward(state, action, s) 
            return reward
   
        successor_states = [s[0] for s in successors]
        if new_state not in successor_states:
            return self.invalid_reward

        return -1

    def do_transition(self, state, action):
        if self.deterministic:
            self.current_state = self._take_deterministic_action(state, action)
            return self.current_state
        else:
            transition_probs = self.get_transition_probs(state, action)
            act_prob = np.random.random()
            cum_prob = 0
            for (s, p) in transition_probs:
                cum_prob += p
                if act_prob <= cum_prob:
                    self.current_state = s
                    return self.current_state

    def at_terminal(self, state=None):
        if state is None:
            state = self.current_state
        return state == self.terminal_state or state == self.terminal_state2

    def at_goal(self, state=None):
        if state is None:
            state = self.current_state
        return state == self.goal_state
