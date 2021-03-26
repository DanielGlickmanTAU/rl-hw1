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


def q3():
    weights = numpy.random.uniform(low=-1., high=1., size=4)
    print(do_episode(weights))


def q4():
    best_weights, max_reward, i = random_search()
    print('best weights', best_weights)
    print('reward', max_reward)


def random_search():
    all_weights = [numpy.random.uniform(low=-1., high=1., size=4) for i in range(10000)]
    max_reward = 0.
    best_weights = None
    for i, w in enumerate(all_weights):
        reward = do_episode(w)
        if reward > max_reward:
            best_weights, max_reward = w, reward
        if reward >= 200.:
            break
    return best_weights, max_reward, i


q3()
q4()

env.close()
