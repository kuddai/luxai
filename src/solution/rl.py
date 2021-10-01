import tensorflow as tf

from solution.utils import print


@tf.function
def n_step_return(states, actions, rewards, is_not_done, *, gamma, n, online_network, target_network):
    #print('Tracing n_step_return')
    batch_size = actions.shape[0]
    steps = actions.shape[1]

    assert steps == n + 1

    last_state = tuple([state[:, -1] for state in states])
    last_actions_mask = last_state[5]

    # Get the best action index according to online network
    online_q_values = online_network(last_state)
    masked_q_values = tf.where(last_actions_mask, online_q_values, tf.fill(online_q_values.shape, -1e9))
    best_online_action_index = tf.argmax(masked_q_values, -1)

    # But get the value of the action according to target network
    target_q_values = target_network(last_state)
    last_state_value = tf.gather(target_q_values, best_online_action_index, axis=-1, batch_dims=2)
    last_unit_mask = tf.cast(states[4][:, -1], dtype=tf.float32)

    is_not_done = tf.cast(is_not_done, dtype=tf.float32)
    rewards = rewards * is_not_done

    last_is_not_done = is_not_done[:, -1]
    last_is_not_done = tf.expand_dims(last_is_not_done, 1)

    #print(last_state_value.shape, is_not_done[:, -1].shape, last_unit_mask.shape)
    cumulative_returns = last_state_value * last_is_not_done * last_unit_mask
    cumulative_returns = tf.reduce_sum(cumulative_returns, axis=-1)

    for step in reversed(range(n)):
        cumulative_returns = rewards[:, step] + gamma * cumulative_returns

    return cumulative_returns


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)


    def test():
        def online_network(state):
            return tf.constant([[0, 1, 2]], dtype=tf.float32) + state[0]

        def target_network(state):
            return online_network(state) * -0.75

        states = tuple(tf.repeat(tf.constant([[[1, -2, 4, -3, 0]]], dtype=tf.float32), 5, 0),)
        actions = tf.repeat(tf.constant([[0, 1, 2, 0, 1]], dtype=tf.int32), 5, 0)
        rewards = tf.repeat(tf.constant([[1, -0.3, 4.5, 0, 4]], dtype=tf.float32), 5, 0)

        result = n_step_return(
            states, actions, rewards,
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ],
            gamma=0.99, n=4, online_network=online_network, target_network=target_network
        )
        expected = tf.constant([1, 0.70299995, 5.1134496, 5.1134496, 3.672556]),
        tf.debugging.assert_near(result, expected)

    test()
