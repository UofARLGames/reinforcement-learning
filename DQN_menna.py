
# coding: utf-8

# In[ ]:

#get_ipython().magic('matplotlib inline')

import gym
import itertools
import numpy as np
import os
import random
import sys
import pdb

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from collections import deque, namedtuple


# In[ ]:

env = gym.envs.make("Breakout-v0")


import tensorflow as tf
# In[ ]:

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]


# In[ ]:

class StateProcessor():
    """
    Processes a raw Atari iamges. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, tf.constant(np.array([84, 84],dtype='int32')), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


# In[ ]:

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.train.SummaryWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.merge_summary([
            tf.scalar_summary("loss", self.loss),
            tf.histogram_summary("loss_hist", self.losses),
            tf.histogram_summary("q_values_hist", self.predictions),
            tf.scalar_summary("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


# In[ ]:

# For Testing....

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Estimator(scope="test")
sp = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    # Example observation batch
    observation = env.reset()

    observation_p = sp.process(sess, observation)
    observation = np.stack([observation_p] * 4, axis=2)
    observations = np.array([observation] * 2)

    # Test Prediction
    print(e.predict(sess, observations))

    # Test training step
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print(e.update(sess, observations, a, y))


# In[ ]:

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


# In[ ]:

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


# In[ ]:

def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Lambda time discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    ep_greedy= make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))
    replay_memory= []

    mini_batch_size= 1

    # Initialize Replay Memory
    state= env.reset()
    state= state_processor.process(sess, state)
    input_state= np.stack([state]*4, axis=2)
    input_next_state= input_state
    for i in range(replay_memory_init_size):
        action_probs= ep_greedy(sess, input_state, epsilon_start)
        action= np.random.choice(range(len(VALID_ACTIONS)), p=action_probs)
        next_state, reward, done, _= env.step(VALID_ACTIONS[action])
        env.render()
        next_state= state_processor.process(sess, next_state)
        input_next_state[:,:,:-1]= input_next_state[:,:,1:]
        input_next_state[:,:,-1]= next_state
        replay_memory.append((input_state, action, input_next_state, reward, done))
        if done:
            state= env.reset()
            state= state_processor.process(sess, state)
            input_state= np.stack([state]*4, axis=2)
        else:
            input_state= input_next_state

    print('Finished Remplay Memory Initialization')
    episode_rewards= []
    episode_lengths= []

    eps_step= (epsilon_start-epsilon_end)/ float(epsilon_decay_steps)
    epsilons= np.arange(epsilon_end, epsilon_start, eps_step)
    t= 0
    for i in range(num_episodes):
        state= env.reset()
        state= state_processor.process(sess, state)
        input_state= np.stack([state]*4, axis=2)
        input_next_state= input_state
        episode_length= 0
        episode_reward= 0

        while True:
            episode_length += 1
            t += 1
            if t> len(epsilons):
                t= len(epsilons)-1

            # copy parameters from q_estimator to t_estimator
            if t % update_target_estimator_every==0:
                copy_model_parameters(sess, q_estimator, target_estimator)

            # use epsilon greedy to get A_next, S_next
            action_probs= ep_greedy(sess, input_state, epsilons[t])
            action= np.random.choice(range(len(VALID_ACTIONS)), p=action_probs)
            next_state, reward, done, _= env.step(VALID_ACTIONS[action])
            episode_reward += reward
            env.render()
            if done:
                break

            next_state= state_processor.process(sess, next_state)
            input_next_state[:,:,:-1]= input_next_state[:,:,1:]
            input_next_state[:,:,-1]= next_state

            # update replay memory
            if len(replay_memory)== replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append((input_state, action, input_next_state, reward, done))

            #sample from replay memory to input to target estimator
            current_batch= random.sample(replay_memory, mini_batch_size)
            current_states, current_actions,current_next_states, current_rewards, current_done \
                = map( np.array,zip(*current_batch))

            # use T_estimator to get Q_next, 0 coz no mini_bs
            qnext= target_estimator.predict(sess, current_states)

            # compute td error
            td_target= current_rewards + discount_factor*np.max(qnext)

            # update q_estimator
            loss= q_estimator.update(sess, current_states, current_actions, td_target)
            print('Current Loss ', loss)

            input_state= input_next_state


        episode_rewards.append(episode_reward/episode_length)
        episode_lengths.append(episode_length)

    return episode_rewards, episode_lengths
# In[ ]:

tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

# Run it!
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=500,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))

