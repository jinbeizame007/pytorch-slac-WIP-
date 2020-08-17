import argparse

import torch
import torch.optim as optim
import numpy as np
import gym

from models import LatentStateModel, Actor, Critic


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
parser.add_argument('--env_steps', type=int, default=1)
parser.add_argument('--gradient_steps', type=int, default=1)
parser.add_argument('--initial_collect_steps', type=int, default=10000)
parser.add_argument('--initial_model_train_steps', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model_batch_size', type=int, default=32)
parser.add_argument('--sequence_size', type=int, default=4)
parser.add_argument('--model_batch_size', type=int, default=32)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--model_lr', type=float, default=1e-4)
parser.add_argument('--actor_lr', type=float, default=3e-4)
parser.add_argument('--critic_lr', type=float, default=3e-4)
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()


env = gym.make(args.env_name)
action_size = env.action_space.shape[0]

model = LatentStateModel(action_space)
actor = Actor(action_size)
critic = Critic(action_size)

optim_model = optim.Adam(model.parameters(), lr=args.model_lr)
optim_model = optim.Adam(actor.parameters(), lr=args.actor_lr)
optim_model = optim.Adam(critic.parameters(), lr=args.critic_lr)


state = env.reset()

for iteration in range(10000):
    ########################
    ### environment step ###
    ########################
    feature = model.encoder(state)
    action = actor.get_action(feature)

    next_state, reward, done, _ = env.step(action)

    if done:
        state = env.reset()


    #####################
    ### gradient step ###
    #####################

    if iteration < args.initial_collect_steps:
        continue

    if iteration < args.initial_collect_steps + args.initial_model_train_steps:

    
