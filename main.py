import gym

import numpy

should_render = False

env = gym.make('CartPole-v0')


def do_episode(weights):
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
        if done:
            break

    return total_reward


# q
weights = numpy.random.uniform(low=-1., high=1., size=4)
print(do_episode(weights))

env.close()
