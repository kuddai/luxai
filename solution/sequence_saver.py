import numpy as np
import tensorflow as tf


class Sequence:
    def __init__(self, t, state_spec, rnn_state, sequence_length):
        self.is_open = True
        self.start_t = t
        self.t = t
        self.rnn_state = rnn_state
        self.sequence_length = sequence_length
        self.state_spec = state_spec
        self.states = []
        self.additional_data = []

        assert (rnn_state is None) or (len(rnn_state) == 2)

        with tf.device('CPU:0'):
            self.actions = np.zeros((self.sequence_length,), dtype=np.int64)
            self.rewards = np.zeros((self.sequence_length,), dtype=np.float32)
            self.is_not_done = np.zeros((self.sequence_length,), dtype=np.bool)

            for element in tf.nest.flatten(state_spec):
                self.states.append(
                    np.zeros((self.sequence_length,) + element.shape, dtype=element.dtype.as_numpy_dtype)
                )

    def add(self, state, action, reward, additional_data=None):
        i = self.t - self.start_t
        assert i < self.sequence_length

        for k, element in enumerate(tf.nest.flatten(state)):
            self.states[k][i] = element

        self.actions[i] = action
        self.rewards[i] = reward
        self.is_not_done[i] = True

        self.t += 1

        if self.t - self.start_t >= self.sequence_length:
            self.is_open = False

        if additional_data is not None:
            self.additional_data.append(additional_data)

    def __len__(self):
        return self.t - self.start_t

    def to_dict(self):
        return {
            'states': [*map(tf.constant, self.states)],
            'actions': tf.constant(self.actions),
            'rewards': tf.constant(self.rewards),
            'is_not_done': tf.constant(self.is_not_done),
            'rnn_state': self.rnn_state,
            'additional_data': self.additional_data
        }


class SequenceSaver:
    def __init__(self, state_spec, sequence_length, create_sequence_interval):
        self.state_spec = state_spec
        self.sequence_length = sequence_length
        self.create_sequence_interval = create_sequence_interval

        self.t = 0
        self.next_seq_t = 0
        self.seqs = []

    def add(self, state, action, reward, rnn_state, additional_data=None):
        if self.t == self.next_seq_t:
            self.seqs.append(
                Sequence(self.t, self.state_spec, rnn_state, self.sequence_length))
            self.next_seq_t += self.create_sequence_interval

        for seq in reversed(self.seqs):
            if not seq.is_open:
                break

            seq.add(state, action, reward, additional_data)

        self.t += 1

    def close(self):
        self.t = 0
        self.next_seq_t = 0

        for seq in reversed(self.seqs):
            if not seq.is_open:
                break

            seq.is_open = False

    def get(self):
        return [s.to_dict() for s in self.seqs]


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    def test():
        state_spec = (
            tf.TensorSpec((2, 3), dtype=tf.float32),
            (
                tf.TensorSpec(tuple(), dtype=tf.float32),
                tf.TensorSpec((1,), dtype=tf.float32),
            )
        )

        def gen():
            return (
                tf.random.uniform((2, 3)),
                (
                    tf.random.uniform(tuple()),
                    tf.random.uniform((1,))
                )
            )
        seq = SequenceSaver(state_spec, 4, 2)

        for i in range(5):
            seq.add(
                gen(),
                tf.random.uniform(tuple(), 0, 10, dtype=tf.int64),
                tf.random.uniform(tuple(), 0, 5),
                None
            )

        result = seq.get()
        assert len(result) == 3
        assert all(result[0]['is_not_done'])
        assert len([*filter(None, result[1]['is_not_done'])]) == 3
        assert len([*filter(None, result[2]['is_not_done'])]) == 1
        for element in result:
            for i, is_not_done in enumerate(element['is_not_done']):
                if is_not_done:
                    continue

                for state_element in element['states']:
                    assert tf.math.count_nonzero(state_element[i]) == 0

                assert element['rewards'][i] == 0
                assert element['actions'][i] == 0

    test()
