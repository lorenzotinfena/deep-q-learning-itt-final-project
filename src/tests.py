from collections import deque

class ReplayMemory():
    def __init__(self, max_size):
        self._max_size = max_size
        self._buffer = deque(maxlen=max_size)

r = ReplayMemory(1)