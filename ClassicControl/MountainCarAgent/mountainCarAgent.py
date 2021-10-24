import time
import matplotlib.pyplot as plt
from numpy import cos
import gym
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from ClassicControl.MyDQLAgent import DQLAgent
from ClassicControl.visualizer import clean, plot


def main():
    name = 'MountainCar-v0'
    env = gym.make(name)
    episodes = 1000
    model_path = 'MountainCarAgent'

    start_training(env, model_path, episodes)

    result_path = 'MountainCarAgentResults.csv'
    call_test_model(env, model_path, episodes, result_path)
    plot_data(episodes, result_path)
    plt.savefig('MountainCarAgent.png')
    plt.show()


def call_test_model(env, model_path, episodes, result_path):
    df = pd.DataFrame()
    for i in range(4):
        modelStart = time.perf_counter()
        scores = test_model(env, f'{model_path}.{i}.h5', episodes)
        modelEnd = time.perf_counter()
        print(f'Model {model_path}.{i} finished in {modelEnd - modelStart:0.3f} seconds.')
        df[f'{model_path}.{i}'] = scores
    df.to_csv(result_path)


def test_model(env, modelPath, episodes) -> list:
    model = load_model(modelPath)
    action_space = [0, 1, 2]
    scores = []
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        i = 0
        while not done:
            #env.render()
            action = action_space[np.argmax(model.predict(state))]
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            i += 1
        scores.append(-i)
        print(f"Episode: {episode + 1}/{episodes}. Score: {-i}")
    return scores


def start_training(env, path, episodes):
    paths = [f'{path}.{i}' for i in range(4)]
    optimizers = ['Adam'] * 4
    layers = [1, 1, 2, 3]
    densities = [[128]] * 2 + [[128, 64]] + [[256, 128, 64]]
    episodes_nums = []
    time_elapsed = []

    for i in range(4):
        start = time.perf_counter()
        episodes_nums.append(train_model(env, paths[i], episodes, optimizers[i], layers[i], densities[i]))
        end = time.perf_counter()
        time_elapsed.append(end - start)

    print('---------------------------------------------')
    for i in range(4):
        print(f'{paths[i]} finished in {episodes_nums[i]} episodes. Time elapse: {time_elapsed[i]}')
    print('---------------------------------------------')


def train_model(gameEnv, path, episodes, opt, layers, densities):
    agent = DQLAgent(game=gameEnv, state=gameEnv.reset(), action_space=[0, 1, 2],
                     path=path, episodes=episodes, opt=opt, scoreFunc=scoreFunc, rewardFunc=rewardFunc,
                     layers=layers, objective=objective, densities=densities, loseLoss=loseLoss)
    return agent.run()


def objective(agent: DQLAgent):
    s = agent.state[0]
    return s[0] >= 0.5


def loseLoss(agent: DQLAgent):
    if agent.step != agent.env._max_episode_steps - 1:
        return 100


def rewardFunc(agent: DQLAgent):
    s = agent.state[0]
    return s[0]


def scoreFunc(agent: DQLAgent):
    return -agent.step


def plot_data(episodes: int, path: str):
    df = clean(path, episodes)
    plot(df)


if __name__ == '__main__':
    main()