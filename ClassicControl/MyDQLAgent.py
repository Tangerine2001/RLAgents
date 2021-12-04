import random
import tensorflow as tf
from collections import deque
import numpy as np
from ClassicControl.MyModel import myModel


class DQLAgent():
    def __init__(self, game, path: str, activation: str, layers: int, densities, rewardFunc, objective,
                 loseLoss, scoreFunc, episodes=1000, gamma: int = 0.9):
        self.env = game
        self.path = path
        self.objective = objective
        self.loseLoss = loseLoss
        self.scoreFunc = scoreFunc
        self.rewardFunc = rewardFunc

        self.state = self.env.reset()
        self.prev_state = self.state
        self.start_state = self.state
        self.state_size = self.state.shape[0]
        self.action_size = self.env.action_space.n
        self.episodes = episodes
        self.memory = deque(maxlen=10000)
        self.step = 0

        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 512
        self.train_start = 3000

        self.model = myModel(env=self.env, name=path, num_of_layers=layers, densities=densities, activation=activation)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # This is to decay epsilon to focus more on random in the beginning
        # then reinforced decisions toward the end
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.model.predict(state.reshape(1, self.state_size)))

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        samples = np.array(random.sample(self.memory, min([len(self.memory), self.batch_size])))

        state_space = np.array(samples[:, 0].tolist(), dtype=float)
        next_space = np.array(samples[:, 1].tolist(), dtype=float)
        actions = np.array(samples[:, 2].tolist(), dtype=np.int32)
        rewards = np.array(samples[:, 3].tolist(), dtype=float)
        done = np.array(samples[:, 4].tolist(), dtype=bool)

        target = self.model.predict(state_space)
        nextTarget = self.model.predict(next_space)

        for i in range(self.batch_size):
            if done[i]:
                target[i, actions[i]] = rewards[i]
            else:
                target[i, actions[i]] = rewards[i] + self.gamma * np.max(nextTarget[i])

        self.model.fit(state_space, target, batch_size=self.batch_size * 2, verbose=0, epochs=10)

    def save(self, name):
        self.model.save(name)

    def run(self):
        winTimes = 0
        for episode in range(self.episodes):
            self.start_state = self.env.reset()
            done = False
            self.step = 0
            while not done:
                # Comment out the below to skip watching the learning process
                # self.env.render()
                # print(f'Step: {self.step}')

                action = self.act(self.state)
                # _ blocks the output. We don't care about extraneous info here
                next_state, reward, done, _ = self.env.step(action)
                reward = self.rewardFunc(self)

                # Progress the state forward
                self.prev_state = self.state
                self.state = next_state

                self.step += 1

                if done:
                    print(f"Episode: {episode + 1}/{self.episodes}, score: {self.scoreFunc(self)}, e: {self.epsilon:.2}")
                    if self.objective(self):
                        if self.step <= 100:
                            if winTimes == 2:
                                print(f"Saving trained model as {self.path}.h5")
                                self.save(self.path + ".h5")
                                return episode
                            winTimes += 1
                        else:
                            winTimes = 0
                    else:
                        winTimes = 0

                self.remember(self.prev_state, self.state, action, reward, done)

                self.replay()

        print("Unable to finish training in time")
        print(f"Saving trained model as {self.path}.h5")
        self.save(self.path + ".h5")

