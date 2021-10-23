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
    path = 'CartPoleAgent/CartPoleResultsv3.csv'
    model_path = 'CartPole-v1'
    episodes = 1000
    env = gym.make(name)

    # start_training(env, name, model_path, episodes)
    #
    # call_test_model(env, name, episodes, path)
    plot_data(episodes, path)
    plt.savefig('CartPoleAgent/CartPoleResultsv3.png')
    plt.show()


def plot_data(episodes: int, path: str):
    df = clean(path, episodes)
    plot(df)


def start_training(env, name, path, episodes):
    names = [f'{name}.{i}' for i in range(4)]
    episodes_nums = []
    time_elapsed = []

    start = time.perf_counter()
    episodes_nums.append(train_model(env, f'{path}.0', episodes, 'Adam', 1, [128]))
    end = time.perf_counter()
    time_elapsed.append(end - start)

    start = time.perf_counter()
    episodes_nums.append(train_model(env, f'{path}.1', episodes, 'RMSProp', 1, [128]))
    end = time.perf_counter()
    time_elapsed.append(end - start)

    start = time.perf_counter()
    episodes_nums.append(train_model(env, f'{path}.2', episodes, 'Adam', 3, [256, 128, 64]))
    end = time.perf_counter()
    time_elapsed.append(end - start)

    start = time.perf_counter()
    episodes_nums.append(train_model(env, f'{path}.3', episodes, 'RMSProp', 3, [256, 128, 64]))
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
        scores = test_model(env, f'CartPoleAgent/{name}.{i}.h5', episodes)
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
            env.render()
            action = np.argmax(model.predict(state))
            next_state, reward, done, _ = env.step(action)
            state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            i += 1
        scores.append(i)
        print(f"Episode: {episode + 1}/{episodes}. Score: {i}")
    return scores


def train_model(gameEnv, path, episodes, opt, layers, densities):
    agent = DQLAgent(game=gameEnv, state=gameEnv.reset(), action_space=[0, 1],
                     path=path, episodes=episodes, opt=opt, scoreFunc=scoreFunc, rewardFunc=rewardFunc,
                     layers=layers, objective=objective, densities=densities, loseLoss=loseLoss)
    return agent.run()


def objective(agent: DQLAgent):
    return agent.step == agent.env._max_episode_steps


def loseLoss(agent: DQLAgent):
    if agent.step != agent.env._max_episode_step:
        return 100


def rewardFunc(agent: DQLAgent):
    return 1


def scoreFunc(agent: DQLAgent):
    return agent.step


def random_policy(observation, action_space):
    return action_space.sample()


def theta_policy(observation, action_space):
    return 1 if observation[2] > 0 else 0


def omega_policy(observation, action_space):
    return 0 if observation[3] < 0 else 1


if __name__ == "__main__":
    main()
