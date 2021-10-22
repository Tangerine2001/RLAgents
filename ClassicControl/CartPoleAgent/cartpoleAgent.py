import time
import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from ClassicControl.MyDQLAgent import DQLAgent
from ClassicControl.visualizer import clean, plot

def main():
    name = 'CartPole-v1'
    path = 'CartPoleResultsv1.csv'
    episodes = 1000
    env = gym.make(name)

    #start_training(env, name, episodes)
    #call_test_model(env, name, episodes, path)
    #plot_data(episodes, path)
    # plt.show()
    # plt.savefig(f'{path}.png')

def plot_data(episodes: int, path: str):
    df = clean(path, episodes)
    plot(df)


def start_training(env, name, episodes):
    names = [f'{name}.{i}' for i in range(4)]
    episodes_nums = []
    time_elapsed = []

    start = time.perf_counter()
    episodes_nums.append(train_model(env, 'CartPole-v1.0', episodes, 'Adam', 1, [128]))
    end = time.perf_counter()
    time_elapsed.append(end - start)

    start = time.perf_counter()
    episodes_nums.append(train_model(env, 'CartPole-v1.1', episodes, 'RMSProp', 1, [128]))
    end = time.perf_counter()
    time_elapsed.append(end - start)

    start = time.perf_counter()
    episodes_nums.append(train_model(env, 'CartPole-v1.2', episodes, 'Adam', 3, [256, 128, 64]))
    end = time.perf_counter()
    time_elapsed.append(end - start)

    start = time.perf_counter()
    episodes_nums.append(train_model(env, 'CartPole-v1.3', episodes, 'RMSProp', 3, [256, 128, 64]))
    end = time.perf_counter()
    time_elapsed.append(end - start)

    print('---------------------------------------------')
    for i in range(4):
        print(f'{names[i]} finished in {episodes_nums[i]} episodes. Time elapse: {time_elapsed[i]}')
    print('---------------------------------------------')


def call_test_model(env, name, episodes, path):
    df = pd.DataFrame()
    for i in range(4):
        modelStart = time.perf_counter()
        scores = test_model(env, f'{name}.{i}.h5', episodes)
        modelEnd = time.perf_counter()
        print(f'Model {name}.{i} finished in {modelEnd - modelStart:0.3f} seconds.')
        df[f'{name}.{i}'] = scores
    df.to_csv(path)


def test_model(env, modelPath, episodes) -> list:
    model = load_model(modelPath)
    scores = []
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        i = 0
        while not done:
            #env.render()
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            i += 1
        scores.append(i)
        print(f"Episode: {episode + 1}/{episodes}. Score: {i}")
    return scores


def train_model(gameEnv, name, episodes, opt, layers, densities):
    agent = DQLAgent(game=gameEnv, name=name, episodes=episodes, opt=opt,
                     layers=layers, densities=densities, gamma=0.95, loseLoss=100)
    return agent.run()


def random_policy(observation, action_space):
    return action_space.sample()


def theta_policy(observation, action_space):
    return 1 if observation[2] > 0 else 0


def omega_policy(observation, action_space):
    return 0 if observation[3] < 0 else 1


if __name__ == "__main__":
    main()
