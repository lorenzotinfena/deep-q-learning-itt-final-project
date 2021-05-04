from collections import deque
import numpy as np

class ReplayMemory():
    def __init__(self, max_size):
        pass
        #self._max_size = max_size
        #self._buffer = deque(maxlen=self._max_size)
        
    def full(self):
        return len(self._buffer) == self._max_size

    def put(self, state, action, reward, done, next_state):
        self._buffer.append((state, action, reward, done, next_state))
    
    def get(self, batch_size):
        """
        args:
            batch_size: <= max_size
        """
        return np.random.choice(self._buffer, batch_size, replace=False)