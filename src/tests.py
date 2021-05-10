import gym
env = gym.make("CartPole-v1")
env.seed(0)
n1 = []
for _ in range(1):
	n1.append(env.reset())
	done = False
	while not done:
		next , reward, done, _ = env.step(env.action_space.sample())
		n1.append(next)
env.seed(0)
n2 = []
for _ in range(1):
	n2.append(env.reset())
	done = False
	while not done:
		next , reward, done, _ = env.step(env.action_space.sample())
		n2.append(next)
print(n1 == n2)
print(n1)
print(n2)