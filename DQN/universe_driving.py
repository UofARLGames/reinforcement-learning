import gym
import universe  # register the universe environments
import pdb
from universe.spaces.vnc_event import KeyEvent
import numpy

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()

f= open('actions.txt', 'w')
VALID_Actions= [('KeyEvent', 'ArrowUp', True), ('KeyEvent', 'ArrowDown', True), ('KeyEvent', 'ArrowRight', True), ('KeyEvent', 'ArrowLeft', True)]
PAUSE_ACTION= [('KeyEvent', 'p', True), ('PointerEvent', 400, 200, 1) ]

def sample_action():
    c= numpy.random.choice(range(len(VALID_Actions)))
    return [VALID_Actions[c]]

iters=10; i=0
while(i<iters):
  action_n= []
  for ob in observation_n:
      event= sample_action()##event= env.action_space.sample()
      action_n.append(event)

  if observation_n != [None]:
      i += 1

  observation_n, reward_n, done_n, info = env.step(action_n)
  f.write(str(action_n)+'\n')
  env.render()

f.close()

observation_n = env.reset()
#observation_n, reward_n, done_n, info = env.step([[PAUSE_ACTION[0]]])
#env.render()
#observation_n, reward_n, done_n, info = env.step([[PAUSE_ACTION[1]]])
#env.render()

while True:
    action_n= []
    for ob in observation_n:
      event= sample_action()##event= env.action_space.sample()
      action_n.append(event)

    observation_n, reward_n, done_n, info = env.step(action_n)

    env.render()

