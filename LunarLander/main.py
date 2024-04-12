import gym
from Agent import Agent
import numpy as np
import torch as T
from utils import plot_learning


if __name__ == "__main__":
    if T.cuda.is_available():
        print("CUDA is available")
        print(f"Using {T.cuda.get_device_name()}")
    else:
        print("Using CPU")

    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation, x_ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, x = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('episode', i, 'score %.2f' % score, 'average_score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_1.png'
    plot_learning(x, scores, eps_history, filename)