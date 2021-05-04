from collections import deque
import numpy as np

class ReplayMemory():
    def __init__(self, max_size):
        self._max_size = max_size
        self._buffer = deque(maxlen=max_size)
    
    def is_full(self):
        return len(self._buffer) == self._max_size

    def put(self, state, action, reward, done, next_state):
        #self._buffer.append((state, action, reward, done, next_state))
        self._buffer.append((state, action, reward, done, next_state))
    
    def get(self, batch_size):
        """
        args:
            batch_size: <= max_size
        """
        random_indexes = np.random.choice(len(self._buffer), size=batch_size)
        return [self._buffer[random_index] for random_index in random_indexes]