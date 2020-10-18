import pickle
import matplotlib.pyplot as plt
import glob, os
import numpy as np
outdir ="outdir"
file_list = {
    "acrobat":f"{outdir}/Acrobot-v1.plt",
    "bipedal":f"{outdir}/BipedalWalker-v3.plt",
    "cartpole":f"{outdir}/CartPole-v1.plt",
    "lunalander":f"{outdir}/LunarLander-v2.plt",
    "mountaincar":f"{outdir}/MountainCar-v0.plt",
    "pendulum":f"{outdir}/Pendulum-v0.plt",

}

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


fig = plt.figure()

elapse_time_max = 0
for name,file in file_list.items():

    try:
        with open(file, "rb") as fd:
            chkpt = pickle.load(fd)
        elapse_time = chkpt["elapse_time"]
        elapse_time_max = max(elapse_time, elapse_time_max)
    except:
        continue

font = {
        'weight': 'bold',
        'size': 12}
import matplotlib

matplotlib.rc('font', **font)
for name,file in file_list.items():

    try:
        with open(file, "rb") as fd:
            chkpt = pickle.load(fd)
        elapse_time = chkpt["elapse_time"]
        reward_threshold = chkpt["reward_threshold"]
        reward_list = chkpt["reward_list"]
        runtime_list = chkpt["runtime_list"]
        total_steps = chkpt["totall_steps"]

        reward_list = np.array(reward_list)
        reward_list = (reward_list - np.min(reward_list)) / (max(1,reward_threshold[file.split("/")[-1].split(".")[0]] - np.min(reward_list)))
        reward_list = moving_average(reward_list, min(len(reward_list) - 2, 50))
        row = np.ones((int(150*100/3/2),))
        row[:len(reward_list)] = np.array(reward_list[:int(150*100/3/2)])
        plt.plot( np.arange(0, elapse_time_max, elapse_time_max/int(150*100/3/2)), row, label=name)
        # plt.plot( np.arange(0, elapse_time_max/4, elapse_time_max/int(150*100/3/2)), row[:int(len(row)/4)], label=name)
        print(f'[{name}]: Elapse time: {elapse_time}, total steps: {total_steps}')
    except:
        continue
plt.title(outdir)
plt.legend(loc=4)
# plt.title("Density comparing to regular NN")
plt.ylabel("Fitness ")
plt.xlabel("Runtime (sec)")
plt.savefig(f"{outdir}.jpg")
plt.show()


    #
    # plt.ylabel("reward")
    # plt.plot(reward_list)
    # plt.show()
    #
    # plt.ylabel("runtime")
    # plt.plot(runtime_list)
    # plt.show()


