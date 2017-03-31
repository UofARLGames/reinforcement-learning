import gym
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from collections import deque, namedtuple

env = gym.envs.make("Breakout-v0")

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]
target_default_path = os.path.join(os.getcwd(),'models/farhad_dqn/target')
q_default_path = os.path.join(os.getcwd(),'models/farhad_dqn/q')
if not os.path.isdir(target_default_path):
    os.makedirs(target_default_path)

if not os.path.isdir(q_default_path):
    os.makedirs(q_default_path)


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

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None,max_to_keep=100):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        self.step = 0

        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.train.SummaryWriter(summary_dir)

        vars_to_save = {t.name:t for t in tf.trainable_variables() if t.name.startswith(self.scope)}
        self.saver = tf.train.Saver(var_list = vars_to_save, name=self.scope ,max_to_keep=100)

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
        self.summaries = tf.summary.merge_all([
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

    def save(self,sess, path,step=None): 
        mypath=os.path.join(path,self.scope)

        self.saver.save(sess, mypath, global_step=self.step,write_meta_graph=True)
        self.step +=1

        print("\n\n\n\n\n\n\n\n list of last_checkpoints: {}\n\n\n\n\n\n\n".format(self.saver.last_checkpoints))
        #self.saver.set_last_checkpoints_with_time(self.saver.last_checkpoints)
        #self.saver.export_meta_graph()
    def restore(self,sess,path,file_to_recover):
        #path =default_path
        recover=os.path.join(path,file_to_recover)
        print("\n\n\n\n\n\n\n\n list of last_checkpoints: {}\n\n\n\n\n\n\n".format(file_to_recover))
        #files = [i for i in os.listdir(path) ]#if os.path.isfile(os.path.join(path,i)) and \
        #  i.startswith(name) and not i.endswith('meta') and not i.endswith('index')]
        #file_to_restore = os.path.join(path,files[-1])

        #new_saver = tf.train.import_meta_graph(path)
        #new_saver.restore(sess,file_to_restore)
        self.saver.restore(sess,recover)
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

def check_file(path,scope):
    files = os.listdir(path)
    #file_to_restore = [i for i in files if not i.endswith('meta') and not i.endswith('index') and i.startswith(scope)]
    file_to_restore = [i.rstrip('.meta') for i in files if i.endswith('meta')]
    return sorted(file_to_restore,reverse=True)
    


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=5000,
                    replay_memory_init_size=500,
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

    """
    Q estimator function which is our network, gets updated using td error
    then we can define policy using our q estimator to get next action and state
    when also need to produce replay memory(which is s and action pairs)
    target_estimator can be computed using (s,a) , 
    target_estimator = R[t+1] + discount_factor * max(Q[next_state])
    Q[next_state] gets computed using target_estimator
    TD_error --> Q[s,a] += learning_rate * (td_target - Q[s,a]). td_target - Q[s,a]
                ^^^^^^^^^^^^^update Q_Estamitor with TD error^^^^^^^^^^^^^^^^^^^^^^
                TD Error is quadratic, why? I don't know yet. 


    Then Target Estimator model gets updated every n step.

    """
    #latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    target_file_to_recover= check_file(target_default_path,target_estimator.scope)
    if target_file_to_recover:
        print("\n\nRestoring model for Target Estimator from : {} ...\n\n\n".format(target_file_to_recover))
        target_estimator.restore(sess,target_default_path,target_file_to_recover[0])
    
    q_file_to_recover= check_file(q_default_path,q_estimator.scope)
    if q_file_to_recover:
        print("\n\nRestoring model for Target Estimator from : {} ...\n\n\n".format(q_file_to_recover))
        q_estimator.restore(sess,q_default_path,q_file_to_recover[0])
    save_every_step = 100
    q_name= 'q'
    target_name = 'target'
    replay_memory = []
    state = env.reset()
    policy = make_epsilon_greedy_policy(q_estimator,len(VALID_ACTIONS))
    prev_state = state_processor.process(sess, state)
    prev_state = np.stack([prev_state] * 4, axis=2)
    next_state = prev_state
    for i in range(replay_memory_init_size):
        actions_probs  = policy(sess,next_state,epsilon_start) 
        action = np.random.choice(np.arange(len(actions_probs)), p=actions_probs)        
        pre_state,reward,done,_ = env.step(VALID_ACTIONS[action])
        next_state[:,:,:-1] = prev_state[:,:,1:]
        next_state[:,:,-1] = state_processor.process(sess, pre_state)
        replay_memory.append((prev_state,action,reward,next_state,done))
    

    state = env.reset()
    state = state_processor.process(sess, state)
    prev_state = np.stack([state] * 4, axis=2)# we are having 4 state as one snaopshot of state to give it temporal information
    next_state = prev_state
    """
    my_epsilon = [epsilon_start]

    for i in range(num_episodes):
        next_eps = epsilon_start-epsilon_decay_steps*epsilon_start
        if next_eps > 0:
            my_epsilon.append(next_eps)
        else:
            my_epsilon.append(epsilon_end)
    """
    print('Finished Remplay Memory Initialization')
    episode_rewards= []
    episode_lengths= []

    eps_step= (epsilon_start-epsilon_end)/ float(epsilon_decay_steps)
    epsilons= np.arange(epsilon_end, epsilon_start, eps_step)
    t= 0
    step = 0
    for i in range(num_episodes):
        state= env.reset()
        state= state_processor.process(sess, state)
        prev_state= np.stack([state]*4, axis=2)
        next_state= prev_state
        episode_length = 0
        episode_reward = 0

        while True:
            t+=1
            step +=1
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop()
            if t> len(epsilons):
                t = len(epsilon)-1

            # copy parameters from q_estimator to t_estimator
            if t % update_target_estimator_every==0:
                copy_model_parameters(sess, q_estimator, target_estimator)

            actions_probs  = policy(sess,prev_state,epsilons[t]) 
            action = np.random.choice(np.arange(len(actions_probs)), p=actions_probs)        
            pre_state,reward,done,_ = env.step(VALID_ACTIONS[action])
            env.render()
            episode_length +=1

            next_state[:,:,:-1] = prev_state[:,:,1:]
            next_state[:,:,-1] = state_processor.process(sess, pre_state) # this again will be fed to policy to produce next action and states

            replay_memory.append((prev_state,action,reward,next_state,done))
            sample = random.sample(replay_memory,batch_size)
            replay_states, replay_actions, replay_rewards, replay_next_states, replay_done = map( np.array,zip(*sample))

            q_next_values = target_estimator.predict(sess, replay_next_states)
            q_next_max = np.max(q_next_values)
            TD_error = replay_rewards + discount_factor * q_next_max
            loss = q_estimator.update(sess,replay_states,replay_actions,TD_error)
            print("loss is: {}".format(loss))
            #stats.episode_rewards[i] += reward
            #stats.episode_lengths[i] = episode_length
            if step % save_every_step == 1:
                q_estimator.save(sess, q_default_path)
                path=target_estimator.save(sess,target_default_path)
            if done:
                break

            prev_state = next_state
        """
        yield episode_length, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i+1],
            episode_rewards=stats.episode_rewards[:i+1])
        """
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    #outputs.append(episode_length,stats.episode_rewardsr
    env.monitor.close()    
    return episode_rewards,episode_length


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

