# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam

class Net():
  def __init__(self, state_size, action_size, lr):
    self.model = Sequential()
    print(type(self.model))
    self.model.add(Dense(24, input_dim=state_size, activation='relu'))
    self.model.add(Dense(24, activation='relu'))
    self.model.add(Dense(action_size, activation='linear'))
    self.model.compile(loss='mse',
                  optimizer=Adam(learning_rate=lr))