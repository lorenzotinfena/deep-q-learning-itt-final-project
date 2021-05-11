from aa import p
import numpy as np
np.random.seed(0)
p()
np.random.seed(0)
p()
import gym
env = gym.make("CartPole-v1")
np.random.seed(0)
env.seed(0)
n1 = []
for _ in range(1):
	n1.append(env.reset()[0])
	done = False
	while not done:
		next , reward, done, _ = env.step(np.random.randint(0, 2))
		n1.append(next[0])
np.random.seed(0)
env.seed(0)
n2 = []
for _ in range(1):
	n2.append(env.reset()[0])
	done = False
	while not done:
		next , reward, done, _ = env.step(np.random.randint(0, 2))
		n2.append(next[0])
print(n1[0] == n2[0]) # true
print(n1[1] == n2[1]) # true
print(n1 == n2) # sometimes true????

