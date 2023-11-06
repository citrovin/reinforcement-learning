# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 0 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 17th October 2020, by alessior@kth.se

### IMPORT PACKAGES ###
# numpy for numerical/random operations
# gym for the Reinforcement Learning environment
import numpy as np
import gym

from collections import deque
from nn import MLP
import torch
from torch.optim import Adam
import tqdm

## Additional imports not directly from lab
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')
import sys

### CREATE RL ENVIRONMENT ###
# env = gym.make('CartPole-v0')        # Create a CartPole environment
env = gym.make('CartPole-v1', render_mode="human")        # v0 of CartPole is out of date and define rendering here
n = len(env.observation_space.low)   # State space dimensionality
m = env.action_space.n               # Number of actions

buffer = deque(maxlen=20)

print(f'Dimensionality of the state space: n={n}')
print(f'Number of possible actions: m={m}')

# function that randomly samples N elements from this buffer.
def sample_from_buffer(buffer, N):
    """Sample N elements from the buffer"""
    random_choice = []
    for i in range(N):
        random_choice.append(buffer[np.random.choice(len(buffer))-1])

    return random_choice

# Define neural network
input_size = n
hidden_size = 8 # from task
output_size = m
learning_rate = 0.01

# Create neural network
model = MLP(input_size, hidden_size, output_size)
optimizer = Adam(model.parameters(), lr=learning_rate)

### PLAY ENVIRONMENT ###
# The next while loop plays 5 episode of the environment
for episode in tqdm.tqdm(range(5)):
    state = env.reset()[0]                  # Reset environment, returns initial
                                         # state
    done = False                         # Boolean variable used to indicate if
                                         # an episode terminated

    while not done:
        # env.render()                     # Render the environment => not used anymore
                                         # (DO NOT USE during training of the
                                         # labs...)

        # action  = np.random.randint(m)   # Pick a random integer between
                                         # [0, m-1]
        state_tensor = torch.tensor(np.array([state]), requires_grad=False)

        # predict and get the index of the highest value in the tensor
        prediction = model(state_tensor)
        action = prediction.argmax().item()

        # The next line takes permits you to take an action in the RL environment
        # env.step(action) returns 4 variables:
        # (1) next state; (2) reward; (3) done variable; (4) additional stuff
        next_state, reward, done, _, _ = env.step(action)

        # save the transition in the buffer x=(s,a,r,s',d)
        buffer.append((state, action, reward, next_state, done))

        # sample from buffer
        if len(buffer) > 3:
            optimizer.zero_grad()

            sample = sample_from_buffer(buffer, 3)

            z = []
            for s in sample:
                preds = model(torch.tensor(s[0]))
                # z.append(preds.argmax().item()) # get the action
                z.append(preds.max().item()) # get the predicted value

            y = torch.zeros(1,3, requires_grad=True)
            z = torch.tensor(z, requires_grad=True).reshape(1,3)

            loss = torch.nn.functional.mse_loss(z,y)

            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        state = next_state

# Close all the windows
env.close()
