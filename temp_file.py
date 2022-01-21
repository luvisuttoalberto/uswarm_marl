import numpy as np
import matplotlib.pyplot as plt
from auxiliary_functions import learning_rate_adaptive, exploration_rate_adaptive

y = np.zeros(3200*20)
for i in range(3200*20):
    y[i] = exploration_rate_adaptive(i, 0.1, 600*20, 3200*20-100)

fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=20)
ax1.set_title('Exploration rate', fontsize=20)

plt.plot(range(3200*20), y)

# plt.ylim(0, 1)
# plt.yscale("log")
plt.grid()

plt.show()

# y = np.zeros(35000)
# for i in range(35000):
#     y[i] = learning_rate_adaptive(i, 0.05, 6000)
#
# fig = plt.figure(figsize=(40, 15))
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.set_xlabel('episode', fontsize=20)
# ax1.set_title('Exploration rate', fontsize=20)
#
# plt.plot(range(35000), y)
#
# # plt.ylim(0, 1)
# # plt.yscale("log")
# plt.grid()
#
# plt.show()