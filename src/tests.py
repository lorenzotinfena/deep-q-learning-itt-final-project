import gym
import gym.wrappers

env = gym.make('FrozenLake-v0')
obs  = env.reset()
env.acti
print(type(env.action_space.sample()))
print(env.action_space.sample())