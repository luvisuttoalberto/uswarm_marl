import numpy as np
from auxiliary_functions import moving_average
import matplotlib.pyplot as plt
from math import pi

visibility_pipe = 0.6

gamma = 0.9995

epsilon_0 = 0.3

reset_type = "area"

# t_star_lr = 150000
t_star_lr = 24000

# std_dev_measure_pipe = pi/64.

# directory = './data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s/' % (visibility_pipe, gamma, epsilon_0, reset_type)
# directory = './data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_closer/' % (visibility_pipe, gamma, epsilon_0, reset_type)
# directory = './data_constant_recognition_extended_gif_try/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_longer/' % (visibility_pipe, gamma, epsilon_0, reset_type)
# directory = './data_constant_recognition/visibility_%.2f_gamma_%.4f_eps_%.1f_reset_%s_noise_%.2f/' % (visibility_pipe, gamma, epsilon_0, reset_type, std_dev_measure_pipe)
directory = './data_multiple_runs/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/' % (visibility_pipe, gamma, reset_type, t_star_lr)

data_swarm_6_states_1_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 1)
average_visited_sections_1_1 = data_swarm_6_states_1_1["average_fraction_pipe"]
moving_average_average_sections_1_1 = moving_average(average_visited_sections_1_1, 100)

print("1:")
for i in range(len(average_visited_sections_1_1)-2):
    if average_visited_sections_1_1[i] > 0.9 and average_visited_sections_1_1[i+1] > 0.9 and average_visited_sections_1_1[i+2] > 0.9:
        print(i)
        break

data_swarm_6_states_1_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 1)
average_visited_sections_1_2 = data_swarm_6_states_1_2["average_fraction_pipe"]
moving_average_average_sections_1_2 = moving_average(average_visited_sections_1_2, 100)


data_swarm_6_states_1_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 1)
average_visited_sections_1_3 = data_swarm_6_states_1_3["average_fraction_pipe"]
moving_average_average_sections_1_3 = moving_average(average_visited_sections_1_3, 100)


data_swarm_6_states_1_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 1)
average_visited_sections_1_4 = data_swarm_6_states_1_4["average_fraction_pipe"]
moving_average_average_sections_1_4 = moving_average(average_visited_sections_1_4, 100)


data_swarm_6_states_1_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 1)
average_visited_sections_1_5 = data_swarm_6_states_1_5["average_fraction_pipe"]
moving_average_average_sections_1_5 = moving_average(average_visited_sections_1_5, 100)

av_av_final_1 = (moving_average_average_sections_1_1 + moving_average_average_sections_1_2 +moving_average_average_sections_1_3 + moving_average_average_sections_1_4 + moving_average_average_sections_1_5) / 5

data_swarm_6_states_2_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 2)
average_visited_sections_2_1 = data_swarm_6_states_2_1["average_fraction_pipe"]
moving_average_average_sections_2_1 = moving_average(average_visited_sections_2_1, 100)

print("2:")
for i in range(len(average_visited_sections_2_1)-2):
    if average_visited_sections_2_1[i] > 0.9 and average_visited_sections_2_1[i+1] > 0.9 and average_visited_sections_2_1[i+2] > 0.9:
        print(i)
        break


data_swarm_6_states_2_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 2)
average_visited_sections_2_2 = data_swarm_6_states_2_2["average_fraction_pipe"]
moving_average_average_sections_2_2 = moving_average(average_visited_sections_2_2, 100)


data_swarm_6_states_2_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 2)
average_visited_sections_2_3 = data_swarm_6_states_2_3["average_fraction_pipe"]
moving_average_average_sections_2_3 = moving_average(average_visited_sections_2_3, 100)


data_swarm_6_states_2_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 2)
average_visited_sections_2_4 = data_swarm_6_states_2_4["average_fraction_pipe"]
moving_average_average_sections_2_4 = moving_average(average_visited_sections_2_4, 100)


data_swarm_6_states_2_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 2)
average_visited_sections_2_5 = data_swarm_6_states_2_5["average_fraction_pipe"]
moving_average_average_sections_2_5 = moving_average(average_visited_sections_2_5, 100)

av_av_final_2 = (moving_average_average_sections_2_1 + moving_average_average_sections_2_2 +moving_average_average_sections_2_3 + moving_average_average_sections_2_4 + moving_average_average_sections_2_5) / 5

data_swarm_6_states_4_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 4)
average_visited_sections_4_1 = data_swarm_6_states_4_1["average_fraction_pipe"]
moving_average_average_sections_4_1 = moving_average(average_visited_sections_4_1, 100)

print("4:")
for i in range(len(average_visited_sections_4_1)-2):
    if average_visited_sections_4_1[i] > 0.9 and average_visited_sections_4_1[i+1] > 0.9 and average_visited_sections_4_1[i+2] > 0.9:
        print(i)
        break


data_swarm_6_states_4_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 4)
average_visited_sections_4_2 = data_swarm_6_states_4_2["average_fraction_pipe"]
moving_average_average_sections_4_2 = moving_average(average_visited_sections_4_2, 100)


data_swarm_6_states_4_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 4)
average_visited_sections_4_3 = data_swarm_6_states_4_3["average_fraction_pipe"]
moving_average_average_sections_4_3 = moving_average(average_visited_sections_4_3, 100)


data_swarm_6_states_4_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 4)
average_visited_sections_4_4 = data_swarm_6_states_4_4["average_fraction_pipe"]
moving_average_average_sections_4_4 = moving_average(average_visited_sections_4_4, 100)


data_swarm_6_states_4_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 4)
average_visited_sections_4_5 = data_swarm_6_states_4_5["average_fraction_pipe"]
moving_average_average_sections_4_5 = moving_average(average_visited_sections_4_5, 100)

av_av_final_4 = (moving_average_average_sections_4_1 + moving_average_average_sections_4_2 +moving_average_average_sections_4_3 + moving_average_average_sections_4_4 + moving_average_average_sections_4_5) / 5

data_swarm_6_states_8_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 8)
average_visited_sections_8_1 = data_swarm_6_states_8_1["average_fraction_pipe"]
moving_average_average_sections_8_1 = moving_average(average_visited_sections_8_1, 100)

print("8:")
for i in range(len(average_visited_sections_8_1)-2):
    if average_visited_sections_8_1[i] > 0.9 and average_visited_sections_8_1[i+1] > 0.9 and average_visited_sections_8_1[i+2] > 0.9:
        print(i)
        break


data_swarm_6_states_8_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 8)
average_visited_sections_8_2 = data_swarm_6_states_8_2["average_fraction_pipe"]
moving_average_average_sections_8_2 = moving_average(average_visited_sections_8_2, 100)


data_swarm_6_states_8_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 8)
average_visited_sections_8_3 = data_swarm_6_states_8_3["average_fraction_pipe"]
moving_average_average_sections_8_3 = moving_average(average_visited_sections_8_3, 100)


data_swarm_6_states_8_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 8)
average_visited_sections_8_4 = data_swarm_6_states_8_4["average_fraction_pipe"]
moving_average_average_sections_8_4 = moving_average(average_visited_sections_8_4, 100)


data_swarm_6_states_8_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 8)
average_visited_sections_8_5 = data_swarm_6_states_8_5["average_fraction_pipe"]
moving_average_average_sections_8_5 = moving_average(average_visited_sections_8_5, 100)

av_av_final_8 = (moving_average_average_sections_8_1 + moving_average_average_sections_8_2 +moving_average_average_sections_8_3 + moving_average_average_sections_8_4 + moving_average_average_sections_8_5) / 5

data_swarm_6_states_16_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 16)
average_visited_sections_16_1 = data_swarm_6_states_16_1["average_fraction_pipe"]
moving_average_average_sections_16_1 = moving_average(average_visited_sections_16_1, 100)

print("16:")
for i in range(len(average_visited_sections_16_1)-2):
    if average_visited_sections_16_1[i] > 0.9 and average_visited_sections_16_1[i+1] > 0.9 and average_visited_sections_16_1[i+2] > 0.9:
        print(i)
        break


data_swarm_6_states_16_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 16)
average_visited_sections_16_2 = data_swarm_6_states_16_2["average_fraction_pipe"]
moving_average_average_sections_16_2 = moving_average(average_visited_sections_16_2, 100)


data_swarm_6_states_16_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 16)
average_visited_sections_16_3 = data_swarm_6_states_16_3["average_fraction_pipe"]
moving_average_average_sections_16_3 = moving_average(average_visited_sections_16_3, 100)


data_swarm_6_states_16_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 16)
average_visited_sections_16_4 = data_swarm_6_states_16_4["average_fraction_pipe"]
moving_average_average_sections_16_4 = moving_average(average_visited_sections_16_4, 100)


data_swarm_6_states_16_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 16)
average_visited_sections_16_5 = data_swarm_6_states_16_5["average_fraction_pipe"]
moving_average_average_sections_16_5 = moving_average(average_visited_sections_16_5, 100)

av_av_final_16 = (moving_average_average_sections_16_1 + moving_average_average_sections_16_2 +moving_average_average_sections_16_3 + moving_average_average_sections_16_4 + moving_average_average_sections_16_5) / 5


directory = './data_baseline_multiple_runs/visibility_%.2f_gamma_%.4f_reset_%s_t_star_%d/' % (visibility_pipe, gamma, reset_type, t_star_lr)

# baseline_data_swarm_6_states_1_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 1)
# baseline_average_visited_sections_1_1 = baseline_data_swarm_6_states_1_1["average_fraction_pipe"]
# baseline_moving_average_average_sections_1_1 = moving_average(baseline_average_visited_sections_1_1, 100)
#
#
# baseline_data_swarm_6_states_1_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 1)
# baseline_average_visited_sections_1_2 = baseline_data_swarm_6_states_1_2["average_fraction_pipe"]
# baseline_moving_average_average_sections_1_2 = moving_average(baseline_average_visited_sections_1_2, 100)
#
#
# baseline_data_swarm_6_states_1_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 1)
# baseline_average_visited_sections_1_3 = baseline_data_swarm_6_states_1_3["average_fraction_pipe"]
# baseline_moving_average_average_sections_1_3 = moving_average(baseline_average_visited_sections_1_3, 100)
#
#
# baseline_data_swarm_6_states_1_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 1)
# baseline_average_visited_sections_1_4 = baseline_data_swarm_6_states_1_4["average_fraction_pipe"]
# baseline_moving_average_average_sections_1_4 = moving_average(baseline_average_visited_sections_1_4, 100)
#
#
# baseline_data_swarm_6_states_1_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 1)
# baseline_average_visited_sections_1_5 = baseline_data_swarm_6_states_1_5["average_fraction_pipe"]
# baseline_moving_average_average_sections_1_5 = moving_average(baseline_average_visited_sections_1_5, 100)
#
#
# baseline_av_av_final_1 = (baseline_moving_average_average_sections_1_1 + baseline_moving_average_average_sections_1_2 +baseline_moving_average_average_sections_1_3 + baseline_moving_average_average_sections_1_4 + baseline_moving_average_average_sections_1_5) / 5

baseline_data_swarm_6_states_2_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 2)
baseline_average_visited_sections_2_1 = baseline_data_swarm_6_states_2_1["average_fraction_pipe"]
baseline_moving_average_average_sections_2_1 = moving_average(baseline_average_visited_sections_2_1, 100)

print("2:")
for i in range(len(baseline_average_visited_sections_2_1)-2):
    if baseline_average_visited_sections_2_1[i] > 0.9 and baseline_average_visited_sections_2_1[i+1] > 0.9 and baseline_average_visited_sections_2_1[i+2] > 0.9:
        print(i)
        break


baseline_data_swarm_6_states_2_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 2)
baseline_average_visited_sections_2_2 = baseline_data_swarm_6_states_2_2["average_fraction_pipe"]
baseline_moving_average_average_sections_2_2 = moving_average(baseline_average_visited_sections_2_2, 100)


baseline_data_swarm_6_states_2_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 2)
baseline_average_visited_sections_2_3 = baseline_data_swarm_6_states_2_3["average_fraction_pipe"]
baseline_moving_average_average_sections_2_3 = moving_average(baseline_average_visited_sections_2_3, 100)


baseline_data_swarm_6_states_2_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 2)
baseline_average_visited_sections_2_4 = baseline_data_swarm_6_states_2_4["average_fraction_pipe"]
baseline_moving_average_average_sections_2_4 = moving_average(baseline_average_visited_sections_2_4, 100)


baseline_data_swarm_6_states_2_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 2)
baseline_average_visited_sections_2_5 = baseline_data_swarm_6_states_2_5["average_fraction_pipe"]
baseline_moving_average_average_sections_2_5 = moving_average(baseline_average_visited_sections_2_5, 100)

baseline_av_av_final_2 = (baseline_moving_average_average_sections_2_1 + baseline_moving_average_average_sections_2_2 +baseline_moving_average_average_sections_2_3 + baseline_moving_average_average_sections_2_4 + baseline_moving_average_average_sections_2_5) / 5

baseline_data_swarm_6_states_4_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 4)
baseline_average_visited_sections_4_1 = baseline_data_swarm_6_states_4_1["average_fraction_pipe"]
baseline_moving_average_average_sections_4_1 = moving_average(baseline_average_visited_sections_4_1, 100)

print("4:")
for i in range(len(baseline_average_visited_sections_4_1)-2):
    if baseline_average_visited_sections_4_1[i] > 0.9 and baseline_average_visited_sections_4_1[i+1] > 0.9 and baseline_average_visited_sections_4_1[i+2] > 0.9:
        print(i)
        break

baseline_data_swarm_6_states_4_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 4)
baseline_average_visited_sections_4_2 = baseline_data_swarm_6_states_4_2["average_fraction_pipe"]
baseline_moving_average_average_sections_4_2 = moving_average(baseline_average_visited_sections_4_2, 100)


baseline_data_swarm_6_states_4_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 4)
baseline_average_visited_sections_4_3 = baseline_data_swarm_6_states_4_3["average_fraction_pipe"]
baseline_moving_average_average_sections_4_3 = moving_average(baseline_average_visited_sections_4_3, 100)


baseline_data_swarm_6_states_4_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 4)
baseline_average_visited_sections_4_4 = baseline_data_swarm_6_states_4_4["average_fraction_pipe"]
baseline_moving_average_average_sections_4_4 = moving_average(baseline_average_visited_sections_4_4, 100)


baseline_data_swarm_6_states_4_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 4)
baseline_average_visited_sections_4_5 = baseline_data_swarm_6_states_4_5["average_fraction_pipe"]
baseline_moving_average_average_sections_4_5 = moving_average(baseline_average_visited_sections_4_5, 100)

baseline_av_av_final_4 = (baseline_moving_average_average_sections_4_1 + baseline_moving_average_average_sections_4_2 +baseline_moving_average_average_sections_4_3 + baseline_moving_average_average_sections_4_4 + baseline_moving_average_average_sections_4_5) / 5

baseline_data_swarm_6_states_8_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 8)
baseline_average_visited_sections_8_1 = baseline_data_swarm_6_states_8_1["average_fraction_pipe"]
baseline_moving_average_average_sections_8_1 = moving_average(baseline_average_visited_sections_8_1, 100)

print("8:")
for i in range(len(baseline_average_visited_sections_8_1)-2):
    if baseline_average_visited_sections_8_1[i] > 0.9 and baseline_average_visited_sections_8_1[i+1] > 0.9 and baseline_average_visited_sections_8_1[i+2] > 0.9:
        print(i)
        break

baseline_data_swarm_6_states_8_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 8)
baseline_average_visited_sections_8_2 = baseline_data_swarm_6_states_8_2["average_fraction_pipe"]
baseline_moving_average_average_sections_8_2 = moving_average(baseline_average_visited_sections_8_2, 100)


baseline_data_swarm_6_states_8_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 8)
baseline_average_visited_sections_8_3 = baseline_data_swarm_6_states_8_3["average_fraction_pipe"]
baseline_moving_average_average_sections_8_3 = moving_average(baseline_average_visited_sections_8_3, 100)


baseline_data_swarm_6_states_8_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 8)
baseline_average_visited_sections_8_4 = baseline_data_swarm_6_states_8_4["average_fraction_pipe"]
baseline_moving_average_average_sections_8_4 = moving_average(baseline_average_visited_sections_8_4, 100)


baseline_data_swarm_6_states_8_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 8)
baseline_average_visited_sections_8_5 = baseline_data_swarm_6_states_8_5["average_fraction_pipe"]
baseline_moving_average_average_sections_8_5 = moving_average(baseline_average_visited_sections_8_5, 100)

baseline_av_av_final_8 = (baseline_moving_average_average_sections_8_1 + baseline_moving_average_average_sections_8_2 +baseline_moving_average_average_sections_8_3 + baseline_moving_average_average_sections_8_4 + baseline_moving_average_average_sections_8_5) / 5

baseline_data_swarm_6_states_16_1 = np.load(directory + '%d_agents/1/data_for_plots.npz' % 16)
baseline_average_visited_sections_16_1 = baseline_data_swarm_6_states_16_1["average_fraction_pipe"]
baseline_moving_average_average_sections_16_1 = moving_average(baseline_average_visited_sections_16_1, 100)

print("16:")
for i in range(len(baseline_average_visited_sections_16_1)-2):
    if baseline_average_visited_sections_16_1[i] > 0.9 and baseline_average_visited_sections_16_1[i+1] > 0.9 and baseline_average_visited_sections_16_1[i+2] > 0.9:
        print(i)
        break

baseline_data_swarm_6_states_16_2 = np.load(directory + '%d_agents/2/data_for_plots.npz' % 16)
baseline_average_visited_sections_16_2 = baseline_data_swarm_6_states_16_2["average_fraction_pipe"]
baseline_moving_average_average_sections_16_2 = moving_average(baseline_average_visited_sections_16_2, 100)


baseline_data_swarm_6_states_16_3 = np.load(directory + '%d_agents/3/data_for_plots.npz' % 16)
baseline_average_visited_sections_16_3 = baseline_data_swarm_6_states_16_3["average_fraction_pipe"]
baseline_moving_average_average_sections_16_3 = moving_average(baseline_average_visited_sections_16_3, 100)


baseline_data_swarm_6_states_16_4 = np.load(directory + '%d_agents/4/data_for_plots.npz' % 16)
baseline_average_visited_sections_16_4 = baseline_data_swarm_6_states_16_4["average_fraction_pipe"]
baseline_moving_average_average_sections_16_4 = moving_average(baseline_average_visited_sections_16_4, 100)


baseline_data_swarm_6_states_16_5 = np.load(directory + '%d_agents/5/data_for_plots.npz' % 16)
baseline_average_visited_sections_16_5 = baseline_data_swarm_6_states_16_5["average_fraction_pipe"]
baseline_moving_average_average_sections_16_5 = moving_average(baseline_average_visited_sections_16_5, 100)

baseline_av_av_final_16 = (baseline_moving_average_average_sections_16_1 + baseline_moving_average_average_sections_16_2 +baseline_moving_average_average_sections_16_3 + baseline_moving_average_average_sections_16_4 + baseline_moving_average_average_sections_16_5) / 5


fig = plt.figure(figsize=(40, 15))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('episode', fontsize=30)

ax1.tick_params(axis="x", labelsize=30)
ax1.tick_params(axis="y", labelsize=30)

ax1.set_title('Average percentage of visible pipe seen by each agent', fontsize=40)
ax1.plot(range(len(average_visited_sections_1_1))[-av_av_final_1.size:], av_av_final_1, label="1", color = "C0", linestyle = "-", linewidth=2)
ax1.plot(range(len(average_visited_sections_1_1))[-av_av_final_2.size:], av_av_final_2, label="2", color = "C1", linestyle = "-", linewidth=2)
ax1.plot(range(len(average_visited_sections_1_1))[-av_av_final_4.size:], av_av_final_4, label="4", color = "C2", linestyle = "-", linewidth=2)
ax1.plot(range(len(average_visited_sections_1_1))[-av_av_final_8.size:], av_av_final_8, label="8", color = "C3", linestyle = "-", linewidth=2)
ax1.plot(range(len(average_visited_sections_1_1))[-av_av_final_16.size:], av_av_final_16, label="16", color = "C4", linestyle = "-", linewidth=2)
# ax1.plot(range(len(baseline_average_visited_sections_1_1))[-baseline_av_av_final_1.size:], baseline_av_av_final_1, label="1", color = "C0", linestyle = ":")
ax1.plot(range(len(baseline_average_visited_sections_2_1))[-baseline_av_av_final_2.size:], baseline_av_av_final_2, label="2", color = "C1", linestyle = ":")
ax1.plot(range(len(baseline_average_visited_sections_2_1))[-baseline_av_av_final_4.size:], baseline_av_av_final_4, label="4", color = "C2", linestyle = ":")
ax1.plot(range(len(baseline_average_visited_sections_2_1))[-baseline_av_av_final_8.size:], baseline_av_av_final_8, label="8", color = "C3", linestyle = ":")
ax1.plot(range(len(baseline_average_visited_sections_2_1))[-baseline_av_av_final_16.size:], baseline_av_av_final_16, label="16", color = "C4", linestyle = ":")

legend = ax1.legend(fontsize=30, loc='center left', bbox_to_anchor=(1, 0.5))
legend.set_title('Number\n of agents:', prop={'size':30})
# legend = ax1.legend(fontsize=30, loc='center left', bbox_to_anchor=(1, 0.5))
# ax1.legend(fontsize=20, loc='center left', title='n_agents:\n dotted = smaller noise', bbox_to_anchor=(1, 0.5))
# ax1.legend(fontsize=20, loc='center left', title='n_agents:\n dotted = with long lost state', bbox_to_anchor=(1, 0.5))
plt.ylim(0, 1)
plt.grid()



# plt.savefig("%s/%.2f_.png" % (directory, episode), bbox_inches='tight')

plt.show()