def prob_dict(probs):
    assert len(probs) == 4
    assert sum(probs) == 1
    return {'B': probs[0], 'K': probs[1], 'O': probs[2], '-': probs[3]}


transition_probabilities = [
    prob_dict([0.1, 0.325, 0.25, 0.325]),
    prob_dict([0.4, 0, 0.4, 0.2]),
    prob_dict([0.2, 0.2, 0.2, 0.4]),
    prob_dict([1, 0, 0, 0])
]

minimum = 0.001


def most_probable_at_k(k):
    def cost(t, state, action):
        if t == 0 and state != 0:
            return minimum
        if state == 3 and t != k:
            return minimum
        if t == k - 1 and action != '-':
            return minimum

        return transition_probabilities[state][action]

    actions = {0: 'B', 1: 'K', 2: 'O', 3: '-'}
    states = [[0, 1, 2, 3] for i in range(k + 1)]
    J = [[minimum, minimum, minimum, minimum] for i in range(k + 1)]
    J[k] = [minimum, minimum, minimum, 1]
    trajectory = [-1] * k

    for i in range(k - 1, -1, -1):
        for state in states[i]:
            options = [cost(i, states[i][state], actions[a]) * J[i + 1][a] for a in actions]
            J[i][state], trajectory[i] = max(options), options.index(max(options))

    trajectory = [actions[i] for i in trajectory[:-1]]
    return "B" + "".join(trajectory)


print(most_probable_at_k(5))
