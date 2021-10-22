import random
from collections import deque
import numpy as np
from MyModel import myModel

class DQLAgent():
    def __init__(self, game, name: str, opt: str, layers: int, densities, episodes=1000, gamma=0.9, loseLoss=100):
        self.name = name
        self.env = game

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.episodes = episodes
        self.memory = deque(maxlen=2000)
        self.loseLoss = loseLoss

        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        self.model = myModel(input_space=(self.state_size,), action_space=self.action_size,
                             name=name, num_of_layers=layers, densities=densities, opt=opt)

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
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        miniBatch = random.sample(self.memory, min([len(self.memory), self.batch_size]))
        #mBCopy = miniBatch.copy()

        stateSpace = np.zeros((self.batch_size, self.state_size))
        #ssCopy = stateSpace.copy()
        nextStateSpace = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            stateSpace[i] = miniBatch[i][0]
            action.append(miniBatch[i][1])
            reward.append(miniBatch[i][2])
            nextStateSpace[i] = miniBatch[i][3]
            done.append(miniBatch[i][4])

        #ssCopy = mBCopy[:, 0]

        target = self.model.predict(stateSpace)
        nextTarget = self.model.predict(nextStateSpace)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.max(nextTarget[i]))

        self.model.fit(stateSpace, target, batch_size=self.batch_size, verbose=0)

    def save(self, name):
        self.model.save(name)

    def run(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # Comment out the below to skip watching the learning process
                # self.env.render()

                action = self.act(state)
                # _ blocks the output. We don't care about extraneous info here
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                if done and i != self.env._max_episode_steps - 1:
                    reward -= self.loseLoss

                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

                if done:
                    print(f"Episode: {episode}/{self.episodes}, score: {i}, e: {self.epsilon:.2}")
                    if i == 500:
                        print(f"Saving trained model as {self.name}.h5")
                        self.save(self.name + ".h5")
                        return episode
                self.replay()

