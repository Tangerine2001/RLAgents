import random
from collections import deque
import numpy as np
from ClassicControl.MyModel import myModel


class DQLAgent():
    def __init__(self, game, state, action_space, path: str, opt: str, layers: int, densities, rewardFunc, objective,
                 loseLoss, scoreFunc, episodes=1000, gamma: int = 0.9):
        self.env = game
        self.path = path
        self.objective = objective
        self.loseLoss = loseLoss
        self.scoreFunc = scoreFunc
        self.rewardFunc = rewardFunc

        self.state = state
        self.prev_state = state
        self.start_state = state
        self.state_size = state.shape[0]
        self.action_size = len(action_space)
        self.action_space = action_space
        self.episodes = episodes
        self.memory = deque(maxlen=2000)
        self.step = 0

        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        self.model = myModel(input_space=(self.state_size,), action_space=self.action_size,
                             name=path, num_of_layers=layers, densities=densities, opt=opt)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # This is to decay epsilon to focus more on random in the beginning
        # then reinforced decisions toward the end
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.action_space[np.random.choice(self.action_size)]
        else:
            return self.action_space[np.argmax(self.model.predict(state))]

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        miniBatch = random.sample(self.memory, min([len(self.memory), self.batch_size]))

        stateSpace = np.zeros((self.batch_size, self.state_size))

        nextStateSpace = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            stateSpace[i] = miniBatch[i][0]
            action.append(miniBatch[i][1])
            reward.append(miniBatch[i][2])
            nextStateSpace[i] = miniBatch[i][3]
            done.append(miniBatch[i][4])

        target = self.model.predict(stateSpace)
        nextTarget = self.model.predict(nextStateSpace)

        for i in range(self.batch_size):
            if done[i]:
                target[i][self.action_space.index(action[i])] = reward[i]
            else:
                target[i][self.action_space.index(action[i])] = reward[i] + self.gamma * (np.max(nextTarget[i]))

        self.model.fit(stateSpace, target, batch_size=self.batch_size, verbose=0)

    def save(self, name):
        self.model.save(name)

    def run(self):
        justWon = False
        for episode in range(self.episodes):
            self.start_state = self.env.reset()
            self.state = np.reshape(self.start_state, [1, self.state_size])
            done = False
            self.step = 0
            while not done:
                # Comment out the below to skip watching the learning process
                self.env.render()
                # print(f'Step: {self.step}')

                action = self.act(self.state)
                # _ blocks the output. We don't care about extraneous info here
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                reward = self.rewardFunc(self)
                self.prev_state = self.state
                self.state = next_state

                self.step += 1

                if done:
                    print(f"Episode: {episode + 1}/{self.episodes}, score: {self.scoreFunc(self)}, e: {self.epsilon:.2}")
                    if self.objective(self):
                        if self.step <= 100:
                            if justWon:
                                print(f"Saving trained model as {self.path}.h5")
                                self.save(self.path + ".h5")
                                return episode
                            justWon = True
                        else:
                            justWon = False

                    # else:
                    #     reward -= self.loseLoss(self)

                self.remember(self.prev_state, action, reward, self.state, done)

                self.replay()

