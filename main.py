'''
Adding normalization to observation and reward, it is extreme important
'''
import gym
from stable_baselines3.common import monitor
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold,CheckpointCallback,CallbackList,BaseCallback,Optional,StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle
import time






reward_threshold = { "MountainCar-v0":-100,
                     "CartPole-v1": 500,
                    "Acrobot-v1":-67, "BipedalWalker-v3":30,
                    "LunarLander-v2":200, 'Pendulum-v0':-100}

model_def = [64,64]

for task in reward_threshold.keys():
    TASK_NAME = task
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/',
                                             name_prefix='rl_model')
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=100 * 150 /2, verbose=1)
    env = gym.make(TASK_NAME)

    log_dir = "./logs"

    env_m = monitor.Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env_m])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold[TASK_NAME], verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
    callback = CallbackList([callback_max_episodes, eval_callback])
   
    model = A2C('MlpPolicy', env, verbose=1,policy_kwargs=dict(net_arch=model_def))
    st = time.time()
    model.learn(total_timesteps=100 * 150 * 10000, callback=callback)
    elapse_time = time.time() - st

    with open("./outdir/"+TASK_NAME + ".plt", "wb") as fd:
        chkpt = {
            "elapse_time": elapse_time,
            "reward_threshold" : reward_threshold,
            "reward_list" : env_m.get_episode_rewards(),
            "timestep_list": env_m.get_episode_lengths(),
            "runtime_list" : env_m.get_episode_times(),
            "totall_steps": env_m.get_total_steps()
        }
        pickle.dump(chkpt, fd)




# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()