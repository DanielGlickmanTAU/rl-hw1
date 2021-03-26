import gym

import numpy

should_render = True

weights = numpy.random.uniform(low=-1., high=1., size=4)
env = gym.make('CartPole-v0')
env.reset()
# todo check first obser
action = 0
total_reward = 0.
for _ in range(200):
    if should_render:
        env.render()

    obs, reward, done, info = env.step(action)  # take a random action
    total_reward += reward
    action = 1 if (weights @ obs) > 0 else 0
    if done and reward:
        print('WTF')
    if not done and not reward:
        print('????')
    if done:
        break
env.close()
print(total_reward)
