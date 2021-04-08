import gym
import numpy as np
import random
from MEC_env import mec_def
from MEC_env import mec_env
import tensorflow as tf
from tensorflow import keras
import tensorboard
import datetime
import AC_agent
from matplotlib import pyplot as plt
import json


print("TensorFlow version: ", tf.__version__)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
plt.rcParams['figure.figsize'] = (9, 9)
# logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

map_size = 200
agent_num = 4
sensor_num = 30
obs_r = 60
collect_r = 40
speed = 6
max_size = 5
sensor_lam = 1e3


MAX_EPOCH = 5000
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002   # learning rate for critic
GAMMA = 0.85     # reward discount
TAU = 0.8      # soft replacement
BATCH_SIZE = 128
alpha = 0.9
beta = 0.1
Epsilon = 0.2
# random seeds are fixed to reproduce the results
map_seed = 1
rand_seed = 17
up_freq = 8
render_freq = 32
np.random.seed(map_seed)
random.seed(map_seed)
tf.random.set_seed(rand_seed)

params = {
    'map_size': map_size,
    'agent_num': agent_num,
    'sensor_num': sensor_num,
    'obs_r': obs_r,
    'collect_r': collect_r,
    'speed': speed,
    'max_size': max_size,
    'sensor_lam': sensor_lam,

    'MAX_EPOCH': MAX_EPOCH,
    'MAX_EP_STEPS': MAX_EP_STEPS,
    'LR_A': LR_A,
    'LR_C': LR_C,
    'GAMMA': GAMMA,
    'TAU': TAU,
    'BATCH_SIZE': BATCH_SIZE,
    # 'alpha': alpha,
    # 'beta': beta,
    'Epsilon': Epsilon,
    'learning_seed': rand_seed,
    'env_seed': map_seed,
    'up_freq': up_freq,
    'render_freq': render_freq
}

mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size, sensor_lam)
env = mec_env.MEC_MARL_ENV(mec_world, alpha=alpha, beta=beta)

AC = AC_agent.ACAgent(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)

m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
f = open('logs/hyperparam/%s.json' % m_time, 'w')
json.dump(params, f)
f.close()
AC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq)
