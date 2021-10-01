import threading

import tensorflow as tf


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.next_index = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.size = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.total_size = tf.Variable(0, dtype=tf.int64, trainable=False)

        self.lock = threading.RLock()

        self.states = None
        self.actions = None
        self.rewards = None
        self.is_not_done = None
        self.additional_data = None
        self.td_errors = None

    def __len__(self):
        with self.lock:
            return self.size.numpy().item()

    def add(self, chunk):
        states = []
        actions = []
        rewards = []
        is_not_done = []
        td_errors = []
        additional_data = []

        for i, row in enumerate(chunk):
            for k, state in enumerate(row['states']):
                if len(states) <= k:
                    states.append([])

                states[k].append(state)
            actions.append(row['actions'])
            rewards.append(row['rewards'])
            is_not_done.append(row['is_not_done'])
            td_errors.append(row['td_error'])
            additional_data.append(row['additional_data'])

        states = [tf.stack(s) for s in states]
        actions = tf.stack(actions)
        rewards = tf.stack(rewards)
        is_not_done = tf.stack(is_not_done)
        td_errors = tf.stack(td_errors)

        with self.lock:
            if self.states is None:
                # Initial add
                self.states = [
                    tf.Variable(
                        tf.zeros(shape=(self.capacity,) + state.shape[1:], dtype=state.dtype),
                        trainable=False
                    )
                    for state in states
                ]
                self.actions = tf.Variable(
                    tf.zeros(shape=(self.capacity,) + actions.shape[1:], dtype=actions.dtype),
                    trainable=False,
                )
                self.rewards = tf.Variable(
                    tf.zeros(shape=(self.capacity,) + rewards.shape[1:], dtype=rewards.dtype),
                    trainable=False,
                )
                self.is_not_done = tf.Variable(
                    tf.zeros(shape=(self.capacity,) + is_not_done.shape[1:], dtype=is_not_done.dtype),
                    trainable=False,
                )
                self.td_errors = tf.Variable(
                    tf.zeros(shape=(self.capacity,), dtype=tf.float32),
                    trainable=False,
                )
                self.additional_data = []

            indices = tf.range(self.next_index, self.next_index + len(chunk), dtype=tf.int64)
            indices = tf.math.mod(indices, self.capacity)

            for i in range(len(self.states)):
                tf.compat.v1.scatter_update(self.states[i], indices, states[i])

            tf.compat.v1.scatter_update(self.actions, indices, actions)
            tf.compat.v1.scatter_update(self.rewards, indices, rewards)
            tf.compat.v1.scatter_update(self.is_not_done, indices, is_not_done)
            tf.compat.v1.scatter_update(self.td_errors, indices, td_errors)
            # self.additional_data += additional_data

            self.next_index.assign((self.next_index + len(chunk)) % self.capacity)
            self.size.assign(min(self.capacity, self.size + len(chunk)))
            self.total_size.assign_add(len(chunk))

    @tf.function
    def _was_replaced(self, indices, prev_total_size):
        print('tracing was repl')
        if self.total_size - prev_total_size >= self.capacity:
            return tf.ones(indices.shape, dtype=tf.bool)

        prev_pos = tf.cast(prev_total_size % self.capacity, dtype=tf.int64)
        curr_pos = tf.cast(self.next_index, dtype=tf.int64)
        if curr_pos < prev_pos:
            return tf.math.logical_or(indices >= prev_pos, indices < curr_pos)

        return tf.math.logical_and(indices >= prev_pos, indices < curr_pos)

    @tf.function
    def _update(self, prev_total_size, indices, td_errors):
        print('tracing update')
        mask = tf.logical_not(self.was_replaced(indices, prev_total_size))

        indices = tf.boolean_mask(indices, mask)
        td_errors = tf.boolean_mask(td_errors, mask)
        tf.compat.v1.scatter_update(self.td_errors, indices, td_errors)

    @tf.function
    def _get_prioritized(self, batch_size):
        print('tracing get prio')
        eps = 0.001
        alpha = 0.9
        beta = 0.6

        td_errors = self.td_errors[:self.size]
        td_errors = tf.math.pow(td_errors + eps, alpha)

        td_sum = tf.reduce_sum(td_errors)
        probs = td_errors / td_sum
        weights = tf.math.pow(probs * tf.cast(self.size, tf.float32), -beta)
        weights = weights / tf.reduce_max(weights)

        log_probs = tf.expand_dims(tf.math.log(probs), 0)
        indices = tf.squeeze(tf.random.categorical(log_probs, batch_size))

        states = [tf.gather(s, indices) for s in self.states]
        actions = tf.gather(self.actions, indices)
        rewards = tf.gather(self.rewards, indices)
        is_not_done = tf.gather(self.is_not_done, indices)
        td_errors = tf.gather(self.td_errors, indices)
        weights = tf.gather(weights, indices)

        return tf.convert_to_tensor(self.total_size), indices, (states, actions, rewards, is_not_done), td_errors, weights

    def was_replaced(self, indices, prev_total_size):
        with self.lock:
            return self._was_replaced(indices, prev_total_size)

    def update(self, prev_total_size, indices, td_errors):
        with self.lock:
            self._update(prev_total_size, indices, td_errors)

    def get_uniform(self, batch_size):
        with self.lock:
            indices = tf.random.uniform((batch_size,), 0, len(self), dtype=tf.int64)
            states = [tf.gather(s, indices) for s in self.states]
            actions = tf.gather(self.actions, indices)
            rewards = tf.gather(self.rewards, indices)
            is_not_done = tf.gather(self.is_not_done, indices)
            td_errors = tf.gather(self.td_errors, indices)

        return (states, actions, rewards, is_not_done), td_errors

    def get_prioritized(self, batch_size):
        with self.lock:
            return self._get_prioritized(batch_size)


if __name__ == '__main__':
    def test():
        #tf.random.set_seed(1)

        rp = ReplayBuffer(5)
        for i in range(6):
            rp.add([
                {
                    'states': [[tf.random.uniform((5,), -5, 5)]],
                    'actions': [0],
                    'rewards': [i],
                    'is_not_done': [False],
                    'additional_data': [],
                    'td_error': i * 0.1,
                },
                {
                    'states': [[tf.random.uniform((5,), -5, 5)]],
                    'actions': [0],
                    'rewards': [i + 6],
                    'is_not_done': [False],
                    'additional_data': [],
                    'td_error': i * 0.1 + 10,
                },
            ])
            if i == 4:
                ts, indices, (s, a, r, d), _, _ = rp.get_prioritized(3)

        assert len(rp) == 5
        expected = tf.expand_dims(tf.constant([5, 11, 9, 4, 10]), 1)
        tf.debugging.assert_equal(rp.rewards, expected)
        assert rp.next_index == 2
        assert rp.total_size == 12

        print(ts, indices, r)
        print(rp.td_errors)
        rp.update(ts, indices, tf.constant([30.0, 31.0, 32.0]))
        print(rp.td_errors)

    while True:
        test()
        break

