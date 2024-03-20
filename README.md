## Atari Deep Q-Learning Implementation

### Introduction
This project is an implementation of the Deep Q-Learning algorithm on the Atari 2600 game Breakout. The implementation is based on the paper "Human-level control through deep reinforcement learning" by Mnih et al. (2015). The goal of the project is to train an agent to play the game of Breakout at a human level of performance.

### Current Status
The code is currently still in development. It is optimized and tested on the Breakout game. The agent is currently learning to play the game "Breakout", but it is not yet at a human level of performance.
The goal is to train the agent to play the game at a human level of performance.

### Architecture
I am using a simple convolutional neural network (CNN) as the function approximator for the Q-function. The CNN has 3 convolutional layers and 2 fully connected layers. The input to the network is the raw pixel values of the game screen. The output is the Q-values for each action.
The state is formed by using four consecutive frames of the game screen. This is done to capture the motion information in the game. 
Further, I am using an experience replay buffer to store the agent's experiences. This buffer is used to sample random batches of experiences to train the network. This helps in stabilizing the training process and allows for easier reproducibility of results and prior states.

### Training
The model is currently fitted on a GPU. The training process is done using the Adam optimizer with a learning rate of 0.0001. The agent is trained for 20,000 frames. The agent is trained using the epsilon-greedy policy with epsilon starting at 1 and decaying to 0.1 over the course of training. The agent is trained using the Deep Q-Learning algorithm with experience replay.

### Motivation
The motivation behind this project is to learn and implement the Deep Q-Learning algorithm on the Atari 2600 game Breakout. The goal is to train an agent to play the game at a human level of performance. This project is a part of my journey to learn more about reinforcement learning and its applications.

