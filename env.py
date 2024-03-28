import gym
import torch
import torchvision.transforms as tt
from PIL import Image
from typing import Tuple
import numpy as np

class AtariEnv:
    def __init__(self, env_name, render_mode="rgb_array", device="cpu"):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.env_name = env_name
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape
        self.env = gym.make(env_name).unwrapped

        self.env.reset()
        self.current_screen = None
        self.done = False
        self.current_lives = None
        self.device = device
        self.state_transformer = StateTransformer(84, 84)

    def reset(self) -> torch.tensor:
        observation = self.env.reset()
        if not isinstance(observation, np.ndarray):
            observation = observation[0]

        return self.state_transformer.apply_transform(Image.fromarray(observation)).to(self.device)

    def step(self, action: int):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def initialize_human_mode(self):
        self.env = gym.make(self.env_name, render_mode='human')

    def take_action(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        observation, reward, done, info, lives = self.env.step(action)
        obs_tensor = self.state_transformer.apply_transform(Image.fromarray(observation)).to(self.device)
        return (obs_tensor, reward, done)

    def play_with_q_model(self, q_model: torch.nn.Module, num_episodes: int):
        q_model.eval()
        q_model = q_model.to("cpu")
        self.initialize_human_mode()

        for episode in range(num_episodes):
            observation = self.reset()
            self.env.render()
            done = False
            ind = 0
            last_four_frames = torch.stack([observation, observation, observation, observation], dim=0)
            while not done:
                with torch.no_grad():
                    action = q_model(last_four_frames).argmax().item()
                observation, reward, done = self.take_action(action)
                last_four_frames[-1] = observation
                ind = ind + 1
                if done:
                    break

class StateTransformer:
    def __init__(self, img_height, img_width):
        self.transforms = tt.Compose([
            tt.Resize((img_height, img_width)),
            tt.Grayscale(),
            tt.ToTensor()
        ])

    def apply_transform(self, state: Image):
        return self.transforms(state).squeeze(0)