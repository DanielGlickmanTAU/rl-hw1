import gym

import numpy

weights = numpy.random.uniform(low=-1., high=1., size=4)
env = gym.make('CartPole-v0')
env.reset()
#todo check first obser
action = 0
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(action)  # take a random action
    action = 1 if (weights @ obs) > 0 else 0
    print(done)
env.close()
