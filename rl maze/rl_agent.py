import numpy as np
import sys
import os
import gym
import time
import kuimaze
from kuimaze import keyboard

# MAP = 'maps/normal/normal3.bmp'
MAP = "maps/easy/easy9.bmp"
MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), MAP)
# PROBS = [0.8, 0.1, 0.1, 0]
PROBS = [1, 0, 0, 0]
GRAD = (0, 0)
keyboard.SKIP = False
VERBOSITY = 2

def learn_policy(env):
    alpha, gamma, epsilon, num_episodes = 0.1, 0.9, 0.1, 2000
    start_time = time.time()
    # Initialize Q-table with zeros
    q_table = np.zeros([env.observation_space.spaces[0].n, env.observation_space.spaces[1].n, env.action_space.n])

    for episode in range(num_episodes):
        # Reset the environment for each episode
        state = env.reset()[0:2]
        if(time.time()-start_time>19.5):
            break
        is_done = False
        while not is_done:
            if(time.time()-start_time>19.5):
                break
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1], :])

            # Take a step in the environment
            next_state, reward, is_done, _ = env.step(action)
            next_state = next_state[0:2]

            # Q-value update
            q_table[state[0], state[1], action] += alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1], :]) - q_table[state[0], state[1], action])
            
            # Update current state
            state = next_state
    # Extract policy from Q-table
    policy = {}
    for i in range(env.observation_space.spaces[0].n):
        for j in range(env.observation_space.spaces[1].n):
            policy[(i, j)] = np.argmax(q_table[i, j, :])
    return policy


# def get_visualisation(table):
#     ret = []
#     for i in range(len(table[0])):
#         for j in range(len(table)):
#             ret.append(
#                 {
#                     "x": j,
#                     "y": i,
#                     "value": [
#                         table[j][i][0],
#                         table[j][i][1],
#                         table[j][i][2],
#                         table[j][i][3],
#                     ],
#                 }
#             )
#     return ret

# if __name__ == "__main__":
#     # Initialize the maze environment
#     env = kuimaze.HardMaze(map_image=MAP, probs=PROBS, grad=GRAD)

#     if VERBOSITY > 0:
#         print("====================")
#         print("works only in terminal! NOT in IDE!")
#         print("press n - next")
#         print("press s - skip to end")
#         print("====================")

#     """
#     Define constants:
#     """
#     # Maze size
#     x_dims = env.observation_space.spaces[0].n
#     y_dims = env.observation_space.spaces[1].n
#     maze_size = tuple((x_dims, y_dims))

#     # Number of discrete actions
#     num_actions = env.action_space.n
#     # Q-table:
#     q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)
#     if VERBOSITY > 0:
#         env.visualise(get_visualisation(q_table))
#         env.render()
#     learn_policy(env)

#     if VERBOSITY > 0:
#         keyboard.SKIP = False
#         env.visualise(get_visualisation(q_table))
#         env.render()
#         keyboard.wait_n_or_s()

#         env.save_path()
#         env.save_eps()
