import tensorflow as tf
import argparse
import pickle
from pdb import set_trace
from mdp import MNISTMDP
from datagen import DataGenerator
from replay_buffer import ReplayBuffer
from util import *

class SuccessorNetwork(object):

    def __init__(self, num_train_episodes, deterministic=True, \
            generate_new=False, path=''):
        self.num_train_episodes = num_train_episodes
        self.deterministic = deterministic
        self.generate_new_episodes = generate_new
        self.path = path
        self.mdp = MNISTMDP(NUM_DIGITS, self.deterministic)
        self.num_actions = self.mdp.get_num_actions()
        self.get_train_data()
        self.get_test_data()
        self.create_inputs_compute_graph()

    def get_train_data(self):
        self.datagen = DataGenerator(self.mdp)
        if self.generate_new_episodes:
            self.train_episodes = self.datagen.gen_episodes( \
                self.num_train_episodes, self.path)
        else:
            self.train_episodes = np.load(self.path)
        self.train_images = np.array([ep[0] for ep in self.train_episodes])
        self.train_actions = np.array([ep[1] for ep in self.train_episodes])
        self.train_reward_labs = np.array([ep[2] for ep in self.train_episodes])
        self.train_qval_labs = np.array([ep[3] for ep in self.train_episodes])
        self.train_label_to_im_dict = self.datagen.label_to_im_dict

    def get_test_data(self):
        self.test_data = load_mnist_test('data/')
        self.test_images = np.array(self.test_data[0])
        self.test_labels = np.array(self.test_data[1])
        self.test_label_to_im_dict = mnist_label_to_image(self.test_data)

    def create_inputs_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, 28, 28, NUM_DIGITS), \
            name='inputs')
        self.actions_raw = tf.placeholder(tf.int64, (None), name='actions_raw')
        self.actions = tf.one_hot(self.actions_raw, self.num_actions, name='actions')
        self.actions.set_shape((None, self.num_actions))
        self.qval_labels = tf.placeholder(tf.float32, (None), name='qval_labels')
        self.reward_labels = tf.placeholder(tf.float32, (None), name='reward_labels')
        self.psi_labels = tf.placeholder(tf.float32, (None, 64), name='psi_labels')
       
    def create_compute_graph(self, scope='SRNet'):
        net = {}
        with tf.variable_scope(scope):
            net['conv1'] = conv_layer(self.inputs, 32, 5, 4)
            net['conv2'] = conv_layer(net['conv1'], 64, 4, 2)
            net['conv3'] = conv_layer(net['conv2'], 64, 3, 1)
            net['conv3'] = tf.contrib.layers.flatten(net['conv3'])
            net['fc1'] = fc_layer(net['conv3'], 64)

            net['fc2'] = fc_layer(self.actions, 64)

            net['concat1'] = tf.concat([net['fc1'], net['fc2']], 1)

            net['fc3'] = fc_layer(net['concat1'], 64)
            net['phi_as'] = fc_layer(net['fc3'], 64) # state-action feature

            net['fc5'] = fc_layer(net['concat1'], 64)
            net['psi_as'] = fc_layer(net['fc5'], 64) # successor feature
            

            w = tf.get_variable("w", [64])
            net['reward'] = tf.reduce_sum(tf.multiply(w, net['phi_as']), 1)
            net['qval'] = tf.reduce_sum(tf.multiply(w, net['psi_as']), 1)

        return net

    def serialize(self, lst, name):
        with open(self.dir_name + '/' + name, 'wb') as fp:
            pickle.dump(lst, fp)

    def IL_train(self, lr):
        eprint("DETERMINISTIC PROB IS {}".format(DETERMINISTIC_PROB))
        #self.dir_name = '{}_{}_{}'.format(self.deterministic, \
        #    NUM_EPISODES, DETERMINISTIC_PROB)
        self.dir_name = 'RL_IL'
        mkdir(self.dir_name)
        self.net = self.create_compute_graph('SRNet_IL')
        self.reward_loss = tf.reduce_mean(tf.losses.mean_squared_error( \
            self.reward_labels, \
            self.net['reward'])) 
        self.qval_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.qval_labels, \
            self.net['qval']))
        self.loss = self.reward_loss + self.qval_loss
        bounds = [100000, 300000]
        values = [lr, 1e-4, 1e-5]
        step_op = tf.Variable(0, name='step', trainable=False)
        learn_rate_op = tf.train.piecewise_constant(step_op, bounds, values)
        self.optimizer = tf.train.AdamOptimizer(learn_rate_op)
        self.minimizer = self.optimizer.minimize(self.loss)

        eprint("[debug] About to train")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        BS = 64
        accs = list()
        all_losses = list()
        for epoch in range(30):
            # Let's shuffle the data every epoch
            np.random.seed(epoch)
            np.random.shuffle(self.train_images)
            np.random.seed(epoch)
            np.random.shuffle(self.train_actions)
            np.random.seed(epoch)
            np.random.shuffle(self.train_reward_labs)
            np.random.seed(epoch)
            np.random.shuffle(self.train_qval_labs)
            # Go through the entire dataset once
            losss = []
            for i in range(0, self.train_images.shape[0]-BS+1, BS):
                # Train a single batch
                batch_images, batch_actions_raw, batch_reward_labs, \
                    batch_qval_labs  = \
                        self.train_images[i:i+BS], \
                        self.train_actions[i:i+BS], \
                        self.train_reward_labs[i:i+BS], \
                        self.train_qval_labs[i:i+BS]
                _, loss, net = self.sess.run(
                    [self.minimizer, self.loss, self.net], 
                    feed_dict={self.inputs: batch_images, \
                               self.actions_raw: batch_actions_raw, \
                               self.reward_labels: batch_reward_labs, \
                               self.qval_labels: batch_qval_labs})
                #eprint("predicted is {}".format(predicted))
                #eprint("actual is {}".format(batch_labels_raw))
                #eprint('[MB %3d] L1 norm: %0.3f  \t  Loss: %0.3f'%(epoch, acc, loss))
                losss.append(loss)

            all_losses.append(np.mean(losss))
            self.serialize(all_losses, 'losses')
            if epoch % 20 == 0 and epoch != 0:
                accuracy = self.evaluate_full_test_set()
                accs.append(accuracy)
                self.serialize(accs, 'accs')
                for i in range(1, 9):
                    pred_action_freq = dict()
                    for _ in range(100):
                        pred_action_freq = self.evaluate_naive(i, pred_action_freq)
                    self.serialize(pred_action_freq, \
                        'pred_action_freq_{}_{}'.format(epoch, i))
            eprint('[%3d] Loss: %0.3f '%(epoch,np.mean(losss)))

    def clone_network(self, from_vars, to_vars, soft_update=False):
        assert len(from_vars) == len(to_vars)
        tau = 0.1
        assign_ops = list()
        for (f, t) in zip(from_vars, to_vars):
            if soft_update:
                assign_op = t.assign(tau * f + (1 - tau) * t)
            else:
                assign_op = t.assign(f)
            assign_ops.append(assign_op)
        return assign_ops

    def RL_train(self, num_episodes):
        assert self.deterministic == True
        self.dir_name = 'RL_IL'
        mkdir(self.dir_name)
        #sr_net = self.create_compute_graph('SRNet_RL1') # theta
        sr_net = self.net
        target_net = self.create_compute_graph('SRNet_RL2') # theta_hat
        #sr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #    "SRNet_RL1")
        sr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            "SRNet_IL")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            "SRNet_RL2")
        # Make a clone of theta as target network theta_hat.
        assign_ops = self.clone_network(sr_vars, target_vars)

        # Initialize replay buffer D to size N.
        self.replay_buffer = ReplayBuffer(10000)

        ####### COMPUTE GRAPH GENERATION ######
        self.reward_loss = tf.reduce_mean(tf.losses.mean_squared_error( \
                    self.reward_labels, \
                    sr_net['reward']))
        #set_trace()
        self.psi_loss = tf.reduce_mean(tf.losses.mean_squared_error(
            sr_net['phi_as'] + self.psi_labels, \
            sr_net['psi_as']))
        self.loss = self.reward_loss + self.psi_loss
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.minimizer = self.optimizer.minimize(self.loss, var_list=sr_vars)
         
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(assign_ops)

        eps = 1.0 # exploration probability term.
        accs = list()
        all_losses = list()
        for epoch in range(num_episodes):
            # Initialize an environment with random configuration.
            self.mdp.reset()
            #eprint("[debug] initial state is {}".format(self.mdp.get_current_state()))
            losss = list()
            while not self.mdp.at_terminal():
                # Get agent's observation and internal state s_t from environment.
                _mdp_state = self.mdp.get_current_state()  
                state = self.get_random_train_image(_mdp_state)
                # Compute Q_{s_t, a} = f(s_t,a ; theta) for every action a in state
                # space.
                actions_lst = self.evaluate(state, net=sr_net)
                 
                # With probability eps, select random action a_t, otherwise select
                # a_t = argmax_a Q_{s_t, a}.
                r = np.random.random()
                act = None
                if r <= eps:
                    pos = np.random.randint(len(actions_lst)) 
                    act = actions_lst[pos][0]
                else: 
                    qvals = [a[2] for a in actions_lst]
                    act = actions_lst[np.argmax(qvals)][0]
                # Execute action a_t to obtain immediate reward r_t and next state
                # s_{t + 1}.
                _mdp_new_state = self.mdp.do_transition(_mdp_state, act)
                reward = self.mdp.get_reward(_mdp_state, act, _mdp_new_state)
                # Store transition (s_t, a_t, r_t, s_{t + 1}) in D.
                new_state = self.get_random_train_image(_mdp_new_state)
                self.replay_buffer.store(state, act, reward, new_state, _mdp_new_state)
                # Sample random mini-batch of transition (s_j, a_j, r_j, s_{j + 1})
                # from D.
                if not self.replay_buffer.full():
                    continue
                #self.serialize(self.replay_buffer.get_buffer(), 'replay_buffer_10000')
                #eprint("[debug] Replay buffer is now full. Starting to train")
                all_actions = self.mdp.get_all_actions()
                transitions = self.replay_buffer.sample(32)
                states = [t[0] for t in transitions]
                actions = [all_actions.index(t[1]) for t in transitions]
                true_rewards = [t[2] for t in transitions]
                next_states = [t[3] for t in transitions]
                _mdp_next_states = [t[4] for t in transitions]

                # Compute psi_{s_{j + 1}, a}, psi_{s_{j + 1}, a}, and Q_{s_{j + 1}, a}
                # using thetahat for every transition j and every action a
                next_actions_lst = list()
                for s in next_states:
                    next_actions_lst.append(self.evaluate(s, net=target_net))

                psi_labels = self.get_successor_labels(_mdp_next_states, \
                    next_actions_lst)

                # Perform gradient descent step to update theta.
                _, l = self.sess.run(
                    [self.minimizer, self.loss], 
                    feed_dict={self.inputs: states, \
                               self.actions_raw: actions, \
                               self.reward_labels: true_rewards, \
                               self.psi_labels: psi_labels})
                losss.append(l)
            if len(losss) == 0:
                continue
            # Anneal exploration term.    
            eps = eps * 0.99
            eps = max(eps, 0.1)
            assign_ops = self.clone_network(sr_vars, target_vars, soft_update=True)
            self.sess.run(assign_ops)
            all_losses.append(np.mean(losss))
            self.serialize(all_losses, 'RL_losses')
            eprint('[%3d] Loss: %0.3f '%(epoch, np.mean(losss)))
            if epoch % 10 != 0:
                continue
            accuracy = self.evaluate_full_test_set(net=target_net)
            accs.append(accuracy)
            self.serialize(accs, 'RL_accs')
            self.generate_sample_episode(epoch, target_net)
            for i in range(1, 9):
                pred_action_freq = dict()
                for _ in range(100):
                    pred_action_freq = self.evaluate_naive(i, pred_action_freq, net=target_net)
                self.serialize(pred_action_freq, \
                        'RL_pred_action_freq_{}_{}'.format(epoch, i))

    def generate_sample_episode(self, epoch, net):
        state = self.mdp.get_random_initial_state()
        ct = 0
        while not self.mdp.at_terminal(state):
            im = self.get_random_test_image(state) 
            self.serialize(im, 'sample_episode_{}_{}'.format(epoch, ct))
            actions_lst = self.evaluate(im, net=net)
            qvals = [a[2] for a in actions_lst]
            pred_action = actions_lst[np.argmax(qvals)][0]
            state = self.mdp._take_deterministic_action(state, pred_action)
            ct += 1
          
    def get_successor_labels(self, next_states, next_actions_lst):
        labels = list()
        gamma = 0.99
        for (i, s) in enumerate(next_states):
            if self.mdp.at_terminal(s):
                # Compute gradients that minimize mean squared error between
                # psi_{sj, aj} and phi_{sj, aj}.
                labels.append(np.zeros((64)))
            else:
                # Compute gradients that minimize mean squared error between
                # psi_{sj, aj} and phi_{sj, aj} + gamma * psi_{s_{j + 1}, a'}
                # where a' = argmax_a Q_{s_{j + 1}, a}.
                actions_lst = next_actions_lst[i] 
                qvals = [a[2] for a in actions_lst]
                best_psi = actions_lst[np.argmax(qvals)][4].reshape((64))
                labels.append(gamma * best_psi)
        return labels

    def get_random_train_image(self, nums):
        final_im = np.zeros((28, 28, NUM_DIGITS))
        for j in range(NUM_DIGITS):
            pos = np.random.randint(len(self.train_label_to_im_dict[nums[j]]))
            tmp_im = self.train_label_to_im_dict[nums[j]][pos].reshape((28, 28))
            final_im[:, :, j] = tmp_im
        return final_im

    def get_random_test_image(self, nums):
        final_im = np.zeros((28, 28, NUM_DIGITS))
        for j in range(NUM_DIGITS):
            pos = np.random.randint(len(self.test_label_to_im_dict[nums[j]]))
            tmp_im = self.test_label_to_im_dict[nums[j]][pos].reshape((28, 28))
            final_im[:, :, j] = tmp_im
        return final_im

    def evaluate_full_test_set(self, net=None):
        correct = 0
        num_examples = 0
        for i in range(0, self.test_images.shape[0]-NUM_DIGITS+1, NUM_DIGITS):
            input_im = np.zeros((28, 28, NUM_DIGITS))
            input_labs = list() 
            c = 0
            for j in range(i, i + NUM_DIGITS):
                im = self.test_images[j].reshape((28, 28)) 
                input_im[:, :, c] = im
                c += 1
                input_labs.append(self.test_labels[j])
            if self.mdp.at_terminal(input_labs):
                continue
            actions_lst = self.evaluate(input_im, net=net)
            max_qval = max([s[2] for s in actions_lst])
            predicted_action = None
            for a in actions_lst:
                if a[2] == max_qval:
                    predicted_action = a[0]
            assert predicted_action is not None
            gr_action = self.datagen.policy[tuple(input_labs)]
            if gr_action[1] == predicted_action[1]:
                correct += 1
            num_examples += 1
            #else:
            #    set_trace()
        accuracy = float(correct) / len(self.test_images)
        eprint("Accuracy on full test set: {}".format(accuracy))
        return accuracy

    def evaluate(self, im, action=None, net=None):
        if net is None:
            net = self.net
        actions_lst = list()
        if action is None:
            for i in range(self.num_actions - 1):
                reward, qval, phi_as, psi_as  = self.sess.run(
                    [net['reward'], net['qval'], net['phi_as'], net['psi_as']],
                    feed_dict={self.inputs: [im],
                               self.actions_raw: [i]})
                actions_lst.append((self.mdp.get_all_actions()[i], reward, qval, \
                        phi_as, psi_as))
            return actions_lst
        else:
            reward, qval, phi_as, psi_as = self.sess.run(
                    [net['reward'], net['qval'], net['phi_as'], net['psi_as']],
                    feed_dict={self.inputs: [im],
                               self.actions_raw: [action]})
            actions_lst = (self.mdp.get_all_actions()[action], reward, qval, \
                        phi_as, psi_as)
            return actions_lst
    
    # Mainly for debugging purposes.
    def evaluate_naive(self, num, pred_action_freq=dict(), net=None):
        if net is None:
            net = self.net
        final_im = self.get_random_test_image([num] * NUM_DIGITS) 
        all_actions = self.mdp.get_all_actions()
        qvals = list()
        for i in range(self.num_actions - 1):
            reward, qval = self.sess.run(
                [net['reward'], net['qval']],
                feed_dict={self.inputs: [final_im],
                           self.actions_raw: [i]})             
            gr_reward = self.mdp.get_reward((num, num), all_actions[i])
            qvals.append(qval) 
        pred_action = all_actions[np.argmax(qvals)]
        if pred_action not in pred_action_freq:
            pred_action_freq[pred_action] = 1
        else:
            pred_action_freq[pred_action] += 1
        return pred_action_freq

def main(deterministic, generate_new, path):
    succ = SuccessorNetwork(NUM_EPISODES, deterministic, generate_new, path)
    succ.IL_train(0.0001)
    succ.RL_train(10000)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VSP')
    parser.add_argument('--deterministic', help="")
    parser.add_argument('--generate_new', help="")
    parser.add_argument('--path', help="")
    parser.add_argument('--deterministic_prob', help="")
    args = parser.parse_args()
    if args.deterministic == 'False':
        args.deterministic = False
    else:
        args.deterministic = True
    if args.generate_new == 'True':
        args.generate_new = True 
    else:
        args.generate_new = False
    DETERMINISTIC_PROB = float(args.deterministic_prob)
    assert args.path != ''
    main(args.deterministic, args.generate_new, args.path)
