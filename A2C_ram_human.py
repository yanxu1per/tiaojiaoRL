
# coding: utf-8

# # Lecture 9: Reinforcement Learning
# In this lecture we will discuss another deep learning technique that has a particular target domain. Reinforcment learning is used when data is unlabeled, especially when decisions need to be made. A great example application of reinforcement learning is creating video game players. In these cases, many decisions must be made (how to move for example) before an outcome is observed (winning or losing). It would be difficult to label each frame, since we don't really know if it's good or bad until the game is over. Reinforcment learning provides a way to evaluate the current state of the game and predict which moves will lead to the best result.

# ![maze](https://media.giphy.com/media/1080OHZwvsMhws/giphy.gif)

# How do you solve a problem in which the interaction between each decision and final outcome isnt clear? In the case of a mouse trapped in a maze, the mouse is trying to decide which way to turn in each of the mazes corridors. Let's try to forumulate the predicament of the mouse.

# ![simplemaze](https://nzmaths.co.nz/sites/default/files/images/uploads/users/3/mazesm.PNG)

# Here we have a simplified view of the maze. Each step the mouse takes moves him one square, and in each square he can choose to move in one of four directions. Of course, his goal is to reach the cheese.
# 
# ## State:
# The first thing the mouse needs to keep track of is where he is, where he's been, and where he's going. All of this information can be modeled by the square the mouse is in. Thus, by keeping track of his current square, the mouse has a good amount of information about his situation.
# 
# ## Action:
# At each step, the mouse much choose which direction to proceed. This choice is an __action__ that transitions him from one __state__ to another.
# 
# ## Reward:
# The mouse must have a goal that hes trying to achieve and a way of knowing that hes achieved it. In this case, it's the cheese at the end of the maze.
# 
# Let's assume that the maze has a time limit. Each attempt the mouse makes to move through the maze is called an __episode__. The episode has a high reward if he makes it to the cheese and a low reward if he doesn't. As the mouse repeats this process a few times, he'll start to learn how to tell if he's in a good place. In other words, he'll learn the __value__ of states in the maze.
# 
# ## Value:
# In this case, the value of a state is the potential it has to reach the reward. If the mouse recognizes a particular corner, he knows he's much more likely to get the reward!

# It's clear how the mouse can improve the quality of his path. As he learns the value of the various squares, he'll figure out the best path to take. The goal of reinforcement learning is to emulate the though process of the rat.
# 
# Historically, the method used to do this was called __Q learning__. Mathematically, the goal is to learn a function $Q(s,a)$ that finds the value of a particular action $a$ given the state $s$. Another way of looking at this is that the function Q is trying to become a __critic__ of the game being played. Rather than focus on making actions, a Q network describes how good a position the player is in. 
# 
# The alternative to Q learning is called policy learning. In this case, the network focuses on simply making a move given the current state. It doesn't really care about value so much as just deciding which move to make right now. This is much more similar to the types of networks we're used to, with the network producing 1 probability for each move. This type of approach can be thought of as an intuitive __player__ or __actor__, who looks at a game and makes a move.
# 
# For a long time, reinforcement learning has been extremely difficult. Using either of the above approaches, researchers found that training RL models was incredibly unstable. It took a few special techniques to Q learning to a point where it was fairly reliable. However, Deep Q Learning (DQN) was still an eyesore, difficult to debug, and very slow to train. Fortunately, the state of the art has made it obsolete!

# ## Actor Critic Interaction
# It turns out that it's actually useful to have both a critic and a player for games. The critic is really good at determinining the potential of a specific state, but not great at realizing that potential. Players are good at making decisions but sometimes lose sight of the bigger picture.
# 
# The idea of combining the two is that a critic will report how good a specific situation is, then when the actor makes his decisions, he can look at how well he did compared to what the critic expected. If he did better, then great! He learns to keep doing something similar. If he underperformed, he knows that he needs to improve. Similarly, because the critic never actually plays the game, he needs to update his expectations based on how the actor does. In this sense, the two are dueling and learning from the other's evaluations.
# 
# This technique is called the Advantage Actor Critic (A2C) technique and was recently developed by google. It's been shown to outperform DQN in all measures.

# ![twitch](http://jwfromm.com/GIX513/images/twitchsei.png)

# [Cool way of testing reinforcement learning techniques](https://gym.openai.com/envs/)

# [Try Some Atari Games](https://www.atari.com/arcade#!/arcade/atari-promo)

# ## RL in action
# To demonstrate reinforcement learning, we're going to implement an A2C network that learns how to play the classic atari game Asteroids

# In[1]:


import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import os
import random
import numpy as np
from IPython import display
import gym
import math
import time
from time import sleep
import mxnet.ndarray as F
import itertools as it

import matplotlib.pyplot as plt
import pdb

#!pip install gym[atari]


# In[2]:


EPISODES = 2020  # Number of episodes to be played
LEARNING_STEPS = 600  # Maximum number of learning steps within each episodes
DISPLAY_COUNT = 10  # The number of episodes to play before showing statistics.


# In[3]:


#  https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
gamma = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.0001
momentum_param = 0.05
learning_rates = [0.0001, 0.01]


# In[24]:


# Other parameters
frame_repeat = 4

ctx = mx.cpu()


# In[5]:


#env_name = 'AssaultNoFrameskip-v4' # Set the desired environment
env_name = "Enduro-ram-v0"
env = gym.make(env_name)
num_action = env.action_space.n # Extract the number of available action from the environment setting

pdb.set_trace()
# In[6]:


# gluon.Block is the basic building block of models.
# You can define networks by composing and inheriting Block:
class Net(gluon.Block):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(16, kernel_size=5, strides=2)
            self.bn1 = gluon.nn.BatchNorm()
            self.conv2 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            self.bn2 = gluon.nn.BatchNorm()
            self.conv3 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            self.bn3 = gluon.nn.BatchNorm()
            #self.lstm = gluon.rnn.LSTMCell(128)
            self.dense1 = gluon.nn.Dense(128, activation='relu')
            self.dense2 = gluon.nn.Dense(64, activation='relu')
            self.action_pred = gluon.nn.Dense(available_actions_count)
            self.value_pred = gluon.nn.Dense(1)
        #self.states = self.lstm.begin_state(batch_size=1, ctx=ctx)

    def forward(self, x):
        x = nd.relu(self.bn1(self.conv1(x)))
        x = nd.relu(self.bn2(self.conv2(x)))
        x = nd.relu(self.bn3(self.conv3(x)))
        x = nd.flatten(x).expand_dims(0)
        #x, self.states = self.lstm(x, self.states)
        x = self.dense1(x)
        x = self.dense2(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return mx.ndarray.softmax(probs), values

class Net2(gluon.Block):
    def __init__(self, available_actions_count):
        super(Net2, self).__init__()
        with self.name_scope():
            # self.conv1 = gluon.nn.Conv2D(16, kernel_size=5, strides=2)
            # self.bn1 = gluon.nn.BatchNorm()
            # self.conv2 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            # self.bn2 = gluon.nn.BatchNorm()
            # self.conv3 = gluon.nn.Conv2D(32, kernel_size=5, strides=2)
            # self.bn3 = gluon.nn.BatchNorm()
            #self.lstm = gluon.rnn.LSTMCell(128)
            self.dense1 = gluon.nn.Dense(64, activation='relu')
            self.d1 = gluon.nn.Dropout(0.5)
            self.dense2 = gluon.nn.Dense(16, activation='relu')
            self.d2 = gluon.nn.Dropout(0.5)
            self.action_pred = gluon.nn.Dense(available_actions_count)
            self.value_pred = gluon.nn.Dense(1)
        #self.states = self.lstm.begin_state(batch_size=1, ctx=ctx)

    def forward(self, x):
        # x = nd.relu(self.bn1(self.conv1(x)))
        # x = nd.relu(self.bn2(self.conv2(x)))
        # x = nd.relu(self.bn3(self.conv3(x)))
        # x = nd.flatten(x).expand_dims(0)
        # #x, self.states = self.lstm(x, self.states)
        x = self.dense1(x)
        x = self.d1(x)
        x = self.dense2(x)
        probs = self.action_pred(x)
        values = self.value_pred(x)
        return mx.ndarray.softmax(probs), values
# In[7]:


loss = gluon.loss.L2Loss()
model = Net2(num_action)
model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate,  "beta1": beta1,  "beta2": beta2, "epsilon": epsilon})


# In[8]:


def preprocess(raw_frame):
    raw_frame = nd.array(raw_frame,mx.cpu())
    #raw_frame = nd.reshape(nd.mean(raw_frame, axis = 2),shape = (raw_frame.shape[0],raw_frame.shape[1],1))
    raw_frame = raw_frame.astype(np.float32)/255.
    data = nd.array(raw_frame).as_in_context(ctx)
    data = data.expand_dims(0)
    return data


# In[ ]:


render_image = False

def train():
    print("Start the training!")
    episode_rewards = 0
    final_rewards = 0

    running_reward = 10 
    train_episodes_finished = 0
    train_scores = [0]
    for episode in range(0, EPISODES):
        next_frame = env.reset()
        proper_frame = next_frame
        s1 = preprocess(proper_frame)

        rewards = []
        values = []
        actions = []
        heads = []

        with autograd.record():
            for learning_step in range(LEARNING_STEPS):
                # Converts and down-samples the input image
                prob, value = model(s1)
                # dont always take the argmax, instead pick randomly based on probability
                index, logp = mx.nd.sample_multinomial(prob, get_prob=True)           
                action = index.asnumpy()[0].astype(np.int64)
                # skip frames
                reward = 0
                for skip in range(frame_repeat+1):
                    # do some frame math to make it not all jumpy and weird
                    new_next_frame, rew, done, _ = env.step(action)
                    proper_frame = next_frame + new_next_frame 
                    next_frame = new_next_frame
                    # can render image if we want
                    #renderimage(proper_frame)
                    reward += rew
                #reward = game.make_action(doom_actions[action], frame_repeat)

                isterminal = done
                rewards.append(reward)
                actions.append(action)
                values.append(value)
                heads.append(logp)

                if isterminal:       
                    #print("finished_game")
                    break
                s1 = preprocess(proper_frame) if not isterminal else None
            train_scores.append(np.sum(rewards))
            # reverse accumulate and normalize rewards
            R = 0
            for i in range(len(rewards) - 1, -1, -1):
                R = rewards[i] + gamma * R
                rewards[i] = R
            rewards = np.array(rewards)
            rewards -= rewards.mean()
            rewards /= rewards.std() + np.finfo(rewards.dtype).eps

            # compute loss and gradient
            L = sum([loss(value, mx.nd.array([r]).as_in_context(ctx)) for r, value in zip(rewards, values)])
            final_nodes = [L]
            for logp, r, v in zip(heads, rewards, values):
                reward = r - v.asnumpy()[0, 0]
                # Here we differentiate the stochastic graph, corresponds to the
                # first term of equation (6) in https://arxiv.org/pdf/1506.05254.pdf
                # Optimizer minimizes the loss but we want to maximizing the reward,
                # so use we use -reward here.
                final_nodes.append(logp * (-reward))
            autograd.backward(final_nodes)
        optimizer.step(s1.shape[0])

        if episode % DISPLAY_COUNT == 0:
            train_scores = np.array(train_scores)
            print("Episodes {}\t".format(episode),
                  "Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max(),
                  "actions: ", np.unique(actions, return_counts=True))
            train_scores = []
        if episode % 1000 == 0 and episode != 0:
            model.save_parameters("./data/Enduro-ram-v0.params")
train()            


# In[ ]:

ACTIONS = env.action_space.n
model.load_parameters("./data/Enduro-ram-v0.params")
def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return

    human_agent_action = a
    print(human_agent_action)

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
# In[16]:

#plt.ion()
def renderimage(next_frame):
    plt.imshow(next_frame);
    #plt.show()
    
    plt.pause(0.001)
    plt.close()

def run_episode():
    next_frame = env.reset()
    done = False
    while not done:
        a=human_agent_action
        if not a:
            s1 = preprocess(next_frame)
            prob, value = model(s1)
            index, logp = mx.nd.sample_multinomial(prob, get_prob=True)           
            action = index.asnumpy()[0].astype(np.int64)
            new_next_frame, rew, done, _ = env.step(action)
            proper_frame = next_frame + new_next_frame 
            #next_frame = new_next_frame
            env.render()
            # renderimage(proper_frame)
            next_frame = new_next_frame
            time.sleep(0.05)
        else:
            new_next_frame, rew, done, _ = env.step(a)
            proper_frame = next_frame + new_next_frame 
            # next_frame = new_next_frame
            env.render()
            # renderimage(proper_frame)
            next_frame = new_next_frame
            time.sleep(0.05)


# In[17]:


run_episode()
env.env.close()

