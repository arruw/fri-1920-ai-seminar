import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

class DQNAgent:

    def __init__(self, state_size, action_size):
        self._state_size = state_size
        self._action_size = action_size
        self._memory = deque(maxlen=2000)
        self._gamma = 0.95              # discount rate
        self._epsilon = 1.0             # exploration rate
        self._epsilon_min = 0.01
        self._epsilon_decay = 0.995
        self._learning_rate = 0.001     # learning rate
        self._model = self._build_model() 

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self._state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self._learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self._epsilon:
            return random.randrange(self._action_size)
        act_values = self._model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self._memory) < batch_size:
            return

        batch = random.sample(self._memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = reward + self._gamma * np.amax(self._model.predict(next_state)[0])

            target_f = self._model.predict(state)
            target_f[0][action] = target
            self._model.fit(state, target_f, epochs=1, verbose=0)
        
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

    def load(self, name):
        self._model = load_model(name)

    def save(self, name):
        self._model.save(name)