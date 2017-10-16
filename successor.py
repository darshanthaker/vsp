import tensorflow as tf
import argparse
import pickle
from pdb import set_trace
from mdp import MNISTMDP
from datagen import DataGenerator
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
        self.create_compute_graph()

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

    def get_test_data(self):
        self.test_data = load_mnist_test('data/')
        self.test_images = np.array(self.test_data[0])
        self.test_labels = np.array(self.test_data[1])
        self.test_label_to_im_dict = mnist_label_to_image(self.test_data)

    def serialize(self, lst, name):
        with open(self.dir_name + '/' + name, 'wb') as fp:
            pickle.dump(lst, fp)

    def create_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, 28, 28, NUM_DIGITS), \
            name='inputs')
        self.actions_raw = tf.placeholder(tf.int64, (None), name='actions_raw')
        self.actions = tf.one_hot(self.actions_raw, self.num_actions, name='actions')
        self.actions.set_shape((None, self.num_actions))
        self.qval_labels = tf.placeholder(tf.float32, (None), name='qval_labels')
        self.reward_labels = tf.placeholder(tf.float32, (None), name='reward_labels')
       
        net = {}
        net['conv1'] = conv_layer(self.inputs, 32, 5, 4)
        net['conv2'] = conv_layer(net['conv1'], 64, 4, 2)
        net['conv3'] = conv_layer(net['conv2'], 64, 3, 1)
        net['conv3'] = tf.contrib.layers.flatten(net['conv3'])
        net['fc1'] = fc_layer(net['conv3'], 64)

        net['fc2'] = fc_layer(self.actions, 64)

        net['concat1'] = tf.concat([net['fc1'], net['fc2']], 1)

        net['fc3'] = fc_layer(net['concat1'], 64)
        net['fc4'] = fc_layer(net['fc3'], 64)

        net['fc5'] = fc_layer(net['concat1'], 64)
        net['fc6'] = fc_layer(net['concat1'], 64)
        

        w = tf.get_variable("w", [64])
        net['reward'] = tf.reduce_sum(tf.multiply(w, net['fc4']), 1)
        net['qval'] = tf.reduce_sum(tf.multiply(w, net['fc6']), 1)

        self.net = net

    def serialize(self, lst, name):
        with open(self.dir_name + '/' + name, 'wb') as fp:
            pickle.dump(lst, fp)

    def train(self, lr):
        self.dir_name = '{}_{}_{}'.format(self.deterministic, \
            NUM_EPISODES, DETERMINISTIC_PROB)
        mkdir(self.dir_name)
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
        for epoch in range(100):
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

            if epoch % 1 == 0 and epoch != 0:
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

    def get_random_test_image(self, num):
        final_im = np.zeros((28, 28, NUM_DIGITS))
        for j in range(NUM_DIGITS):
            pos = np.random.randint(len(self.test_label_to_im_dict[num]))
            tmp_im = self.test_label_to_im_dict[num][pos].reshape((28, 28))
            final_im[:, :, j] = tmp_im
        return final_im

    def evaluate_full_test_set(self):
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
            actions_lst = self.evaluate(input_im)
            max_qval = max([s[2] for s in actions_lst])
            predicted_action = None
            for a in actions_lst:
                if a[2] == max_qval:
                    predicted_action = a[0]
            assert predicted_action is not None
            gr_action = self.datagen.policy[tuple(input_labs)]
            if gr_action == predicted_action:
                correct += 1
            num_examples += 1
            #else:
            #    set_trace()
        accuracy = float(correct) / len(self.test_images)
        eprint("Accuracy on full test set: {}".format(accuracy))
        return accuracy

    def evaluate(self, im):
        actions_lst = list()
        for i in range(self.num_actions - 1):
            reward, qval = self.sess.run(
                [self.net['reward'], self.net['qval']],
                feed_dict={self.inputs: [im],
                           self.actions_raw: [i]})
            actions_lst.append((self.mdp.get_all_actions()[i], reward, qval))
        return actions_lst
    
    # Mainly for debugging purposes.
    def evaluate_naive(self, num, pred_action_freq=dict()):
        final_im = self.get_random_test_image(num) 
        all_actions = self.mdp.get_all_actions()
        qvals = list()
        for i in range(self.num_actions - 1):
            reward, qval = self.sess.run(
                [self.net['reward'], self.net['qval']],
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
    succ.train(0.005)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='VSP')
    parser.add_argument('--deterministic', help="")
    parser.add_argument('--generate_new', help="")
    parser.add_argument('--path', help="")
    args = parser.parse_args()
    if args.deterministic == 'False':
        args.deterministic = False
    else:
        args.deterministic = True
    if args.generate_new == 'True':
        args.generate_new = True 
    else:
        args.generate_new = False
    assert args.path != ''
    main(args.deterministic, args.generate_new, args.path)
