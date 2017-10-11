import tensorflow as tf
from pdb import set_trace
from mdp import MNISTMDP, DataGenerator
from util import *

class SuccessorNetwork(object):


    def __init__(self, num_train_episodes):
        self.num_train_episodes = num_train_episodes
        self.mdp = MNISTMDP(NUM_DIGITS)
        self.num_actions = self.mdp.get_num_actions()
        self.get_train_data()
        self.get_test_data()
        self.create_compute_graph()

    def get_train_data(self):
        #datagen = DataGenerator(self.mdp)
        #self.train_episodes = datagen.gen_episodes(self.num_train_episodes)
        self.train_episodes = np.load('episodes_2_10000_random.npy')
        self.train_images = np.array([ep[0] for ep in self.train_episodes])
        self.train_actions = np.array([ep[1] for ep in self.train_episodes])
        self.train_reward_labs = np.array([ep[2] for ep in self.train_episodes])
        self.train_qval_labs = np.array([ep[3] for ep in self.train_episodes])

    def get_test_data(self):
        mnist = load_mnist_test('data/')
        self.test_label_to_im_dict = mnist_label_to_image(mnist)

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

    def train(self, lr):
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
            accs, losss = [], []
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
                self.evaluate(8)
                self.evaluate(2) 
                self.evaluate(9)
                self.evaluate(0)
            eprint('[%3d] Loss: %0.3f '%(epoch,np.mean(losss)))
    
    def evaluate(self, num):
        final_im = np.zeros((28, 28, 2))
        for j in range(2):
            tmp_im = self.test_label_to_im_dict[num][0].reshape((28, 28))
            final_im[:, :, j] = tmp_im
        for i in range(self.num_actions - 1):
            reward, qval = self.sess.run(
                [self.net['reward'], self.net['qval']],
                feed_dict={self.inputs: [final_im],
                           self.actions_raw: [i]})             
            gr_reward = self.mdp.get_reward((num, num), self.mdp.get_all_actions()[i])
            eprint("State ({}, {}) and action {} has reward {} with ground truth reward {} and qval {}".format(num, num, self.mdp.get_all_actions()[i], reward, gr_reward, qval))
             

def main():
    succ = SuccessorNetwork(10000)
    succ.train(0.005)

if __name__=='__main__':
    main()
