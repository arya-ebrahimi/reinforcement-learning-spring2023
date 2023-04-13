import gymnasium as gym
import numpy as np
from gymnasium.wrappers import record_video

from core.policy_iteration import policy_iteration
from core.value_iteration import value_iteration

def play(mode, gamma=0.9):
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='rgb_array')
    env = record_video.RecordVideo(env, video_folder='runs', name_prefix=mode)
    if mode == 'policy_iteration':
        V, pi = policy_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)
    else:
        V, pi = value_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)
        
    done = False
    state, _ = env.reset()
    env.render()
    total_reward = 0
    while not done:
        action = np.argmax(pi[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        # env.render()
    env.close()
    print('total reward is: ', total_reward)
    
if __name__ == '__main__':
    
    play(mode='policy_iteration', gamma=0)
    play(mode='value_iteration', gamma=0)