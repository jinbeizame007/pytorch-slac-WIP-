from collections import deque

import numpy as np


class SequenceBuffer:
    def __init__(self, sequence_size):
        self.sequence_size = sequence_size
        self.states = deque(maxlen=sequence_size)
        self.actions = deque(maxlen=sequence_size)
        self.rewards =  deque(maxlen=sequence_size)
        self.dones = deque(maxlen=sequence_size)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def append(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def get_sequences(self):
        states = list(self.states)
        actions = list(self.actions)
        rewards = list(self.rewards)
        dones = list(self.dones)
        return states, actions[:-1], rewards[:-1], dones[:-1]


class ReplayMemory:
    def __init__(self, action_size, memory_size=100000, sequence_size=8):
        self.memory_size = memory_size
        self.sequence_size = sequence_size
        self.sequence_buffer = SequenceBuffer(sequence_size+1)
        self.states = np.empty((memory_size, sequence_size+1, 3, 64, 64), dtype=np.int8)
        self.actions = np.empty((memory_size, sequence_size, action_size), dtype=np.float32)
        self.rewards = np.empty((memory_size, sequence_size, 1), dtype=np.float32)
        self.dones = np.empty((memory_size, sequence_size, 1), dtype=np.float32)
    
    def append(state, action, reward, dones):
