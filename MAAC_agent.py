import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy.io as sio
import gym
import time
import random
import datetime
import os
import imageio
import glob
import tqdm
import json

# tf.random.set_seed(11)

# tf.keras.backend.set_floatx('float64')


def discrete_circle_sample_count(n):
    count = 0
    move_dict = {}
    for x in range(-n, n + 1):
        y_l = int(np.floor(np.sqrt(n**2 - x**2)))
        for y in range(-y_l, y_l + 1):
            move_dict[count] = np.array([y, x])
            count += 1
    return (count), move_dict


def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.math.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

# agent actor net: inputs state map,pos,buffer,operation,bandwidth; outputs: move,operation


def agent_actor(input_dim_list, cnn_kernel_size, move_r):
    state_map = keras.Input(shape=input_dim_list[0])
    # position = keras.Input(shape=input_dim_list[1])
    total_buffer = keras.Input(shape=input_dim_list[1])
    done_buffer = keras.Input(shape=input_dim_list[2])
    bandwidth = keras.Input(shape=input_dim_list[3])
    # CNN for map
    cnn_map = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(state_map)
    cnn_map = layers.AveragePooling2D(pool_size=int(input_dim_list[0][0] / (2 * move_r + 1)))(cnn_map)
    cnn_map = layers.AlphaDropout(0.2)(cnn_map)
    move_out = layers.Dense(1, activation='relu')(cnn_map)
    # move_out = move_out / tf.reduce_sum(move_out, [1, 2, 3], keepdims=True)
    # move_out = tf.exp(move_out) / tf.reduce_sum(tf.exp(move_out), [1, 2, 3], keepdims=True)

    # cnn_map = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(cnn_map)
    # cnn_map = layers.MaxPooling2D(pool_size=cnn_kernel_size)(cnn_map)
    # cnn_map = layers.Dropout(0.2)(cnn_map)
    # cnn_output = layers.Flatten()(cnn_map)
    # cnn_output = layers.Dense(128, activation='relu')(cnn_output)
    # move_dist = layers.Dense(move_count, activation='softmax')(move_out)

    # operation
    # total_mlp = layers.Dense(2, activation='relu')(total_buffer)
    # done_mlp = layers.Dense(2, activation='relu')(done_buffer)
    # buffer_mlp = layers.concatenate([total_mlp, done_mlp], axis=-1)
    # bandwidth_in = tf.expand_dims(bandwidth, axis=-1)
    # bandwidth_in = tf.tile(bandwidth_in, [1, 2, 1])
    # # concatenate on dim[1] batch*new*2
    # op_output = layers.concatenate([buffer_mlp, bandwidth_in], axis=-1)

    # op_dist = layers.Dense(input_dim_list[2][1], activation='softmax')(op_output)

    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.transpose(total_mlp, perm=[0, 2, 1])
    exe_op = layers.Dense(input_dim_list[1][1], activation='softmax')(total_mlp)

    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.transpose(done_mlp, perm=[0, 2, 1])
    bandwidth_in = tf.expand_dims(bandwidth, axis=-1)
    bandwidth_in = layers.Dense(1, activation='relu')(bandwidth_in)
    done_mlp = layers.concatenate([done_mlp, bandwidth_in], axis=-1)
    off_op = layers.Dense(input_dim_list[2][1], activation='softmax')(done_mlp)

    op_dist = layers.concatenate([exe_op, off_op], axis=1)
    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, bandwidth], outputs=[move_out, op_dist])
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_aa))
    return model

# center actor net: inputs sensor_map,agent_map,bandwidth_vector; outputs: bandwidth_vec


def center_actor(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(shape=input_dim_list[0])
    pos_list = keras.Input(shape=input_dim_list[1])

    # buffer
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)
    buffer_state = tf.squeeze(buffer_state, axis=-1)

    # pos list
    pos = layers.Dense(2, activation='relu')(pos_list)

    bandwidth_out = layers.concatenate([buffer_state, pos], axis=-1)
    # bandwidth_out = layers.AlphaDropout(0.2)(bandwidth_out)
    bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)
    bandwidth_out = tf.squeeze(bandwidth_out, axis=-1)
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    bandwidth_out = layers.Softmax()(bandwidth_out)
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    # bandwidth_out = bandwidth_out / tf.reduce_sum(bandwidth_out, 1, keepdims=True)
    # bandwidth_out = bandwidth_out / tf.expand_dims(tf.reduce_sum(bandwidth_out, 1), axis=-1)

    model = keras.Model(inputs=[done_buffer_list, pos_list], outputs=bandwidth_out, name='center_actor_net')
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_ca))
    # sensor_map = keras.Input(shape=input_dim_list[0])
    # agent_map = keras.Input(shape=input_dim_list[1])

    # # sensor map:cnn*2
    # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_map)
    # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_cnn)
    # # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # sensor_cnn = layers.Flatten()(sensor_cnn)
    # sensor_cnn = layers.Dense(4, activation='softmax')(sensor_cnn)

    # # agent map
    # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_map)
    # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_cnn)
    # # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # agent_cnn = layers.Flatten()(agent_cnn)
    # agent_cnn = layers.Dense(4, activation='softmax')(agent_cnn)

    # # add bandwidth
    # bandwidth_out = layers.concatenate([sensor_cnn, agent_cnn], axis=-1)
    # bandwidth_out = layers.Dense(input_dim_list[2], activation='softmax')(bandwidth_out)

    # model = keras.Model(inputs=[sensor_map, agent_map], outputs=bandwidth_out, name='center_actor_net')
    # # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_ca))
    return model


# agent aritic net
def agent_critic(input_dim_list, cnn_kernel_size):
    state_map = keras.Input(shape=input_dim_list[0])
    # position = keras.Input(shape=input_dim_list[1])
    total_buffer = keras.Input(shape=input_dim_list[1])
    done_buffer = keras.Input(shape=input_dim_list[2])
    move = keras.Input(shape=input_dim_list[3])
    onehot_op = keras.Input(shape=input_dim_list[4])
    bandwidth = keras.Input(shape=input_dim_list[5])

    # map CNN
    # merge last dim
    map_cnn = layers.Dense(1, activation='relu')(state_map)
    map_cnn = layers.Conv2D(1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
    map_cnn = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(map_cnn)
    map_cnn = layers.AlphaDropout(0.2)(map_cnn)
    # map_cnn = layers.Conv2D(input_dim_list[0][2], kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
    # map_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(map_cnn)
    # map_cnn = layers.Dropout(0.2)(map_cnn)
    map_cnn = layers.Flatten()(map_cnn)
    map_cnn = layers.Dense(2, activation='relu')(map_cnn)

    # mlp
    # pos_mlp = layers.Dense(1, activation='relu')(position)
    band_mlp = layers.Dense(1, activation='relu')(bandwidth)
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.squeeze(total_mlp, axis=-1)
    total_mlp = layers.Dense(2, activation='relu')(total_mlp)
    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.squeeze(done_mlp, axis=-1)
    done_mlp = layers.Dense(2, activation='relu')(done_mlp)

    move_mlp = layers.Flatten()(move)
    move_mlp = layers.Dense(1, activation='relu')(move_mlp)
    onehot_mlp = layers.Dense(1, activation='relu')(onehot_op)
    onehot_mlp = tf.squeeze(onehot_mlp, axis=-1)

    all_mlp = layers.concatenate([map_cnn, band_mlp, total_mlp, done_mlp, move_mlp, onehot_mlp], axis=-1)
    reward_out = layers.Dense(1, activation='relu')(all_mlp)

    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, move, onehot_op, bandwidth], outputs=reward_out)
    # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
    return model


# center critic
def center_critic(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(shape=input_dim_list[0])
    pos_list = keras.Input(shape=input_dim_list[1])
    bandwidth_vec = keras.Input(shape=input_dim_list[2])

    # buffer
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)
    buffer_state = tf.squeeze(buffer_state, axis=-1)
    buffer_state = layers.Dense(1, activation='relu')(buffer_state)
    buffer_state = tf.squeeze(buffer_state, axis=-1)

    # pos list
    pos = layers.Dense(1, activation='relu')(pos_list)
    pos = tf.squeeze(pos, axis=-1)

    # bandvec
    # band_in = layers.Dense(2, activation='relu')(bandwidth_vec)

    r_out = layers.concatenate([buffer_state, pos, bandwidth_vec])
    # r_out = layers.AlphaDropout(0.2)(r_out)
    r_out = layers.Dense(1, activation='relu')(r_out)
    model = keras.Model(inputs=[done_buffer_list, pos_list, bandwidth_vec], outputs=r_out, name='center_critic_net')
    # sensor_map = keras.Input(shape=input_dim_list[0])
    # agent_map = keras.Input(shape=input_dim_list[1])
    # bandwidth_vec = keras.Input(shape=input_dim_list[2])

    # # sensor map:cnn*2
    # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_map)
    # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_cnn)
    # # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # sensor_cnn = layers.Flatten()(sensor_cnn)
    # sensor_cnn = layers.Dense(4, activation='relu')(sensor_cnn)

    # # agent map
    # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_map)
    # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_cnn)
    # # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # agent_cnn = layers.Flatten()(agent_cnn)
    # agent_cnn = layers.Dense(4, activation='relu')(agent_cnn)

    # # add bandwidth
    # bandwidth_out = layers.concatenate([sensor_cnn, agent_cnn, bandwidth_vec], axis=-1)
    # bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)

    # model = keras.Model(inputs=[sensor_map, agent_map, bandwidth_vec], outputs=bandwidth_out, name='center_critic_net')
    # # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
    return model


def update_target_net(model, target, tau=0.8):
    weights = model.get_weights()
    target_weights = target.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * (1 - tau) + target_weights[i] * tau
    target.set_weights(target_weights)


def merge_fl(nets, omega=0.5):
    for agent_no in range(len(nets)):
        target_params = nets[agent_no].get_weights()
        other_params = []
        for i, net in enumerate(nets):
            if i == agent_no:
                continue
            other_params.append(net.get_weights())
        for i in range(len(target_params)):
            others = np.sum([w[i] for w in other_params], axis=0) / len(other_params)
            target_params[i] = omega * target_params[i] + others * (1 - omega)
            # print([others.shape, target_params[i].shape])
        nets[agent_no].set_weights(target_params)


def circle_argmax(move_dist, move_r):
    max_pos = np.argwhere(tf.squeeze(move_dist, axis=-1) == np.max(move_dist))
    # print(tf.squeeze(move_dist, axis=-1))
    pos_dist = np.linalg.norm(max_pos - np.array([move_r, move_r]), axis=1)
    # print(max_pos)
    return max_pos[np.argmin(pos_dist)]


class MAACAgent(object):
    def __init__(self, env, tau, gamma, lr_aa, lr_ac, lr_ca, lr_cc, batch, epsilon=0.2):
        self.env = env
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.index_dim = 2
        self.obs_r = self.env.obs_r
        self.state_map_shape = (self.obs_r * 2 + 1, self.obs_r * 2 + 1, self.index_dim)
        self.pos_shape = (2)
        self.band_shape = (1)
        self.buffstate_shape = (self.index_dim, self.env.max_buffer_size)
        # self.sensor_map_shape = (self.env.map_size, self.env.map_size, self.index_dim)
        # self.agent_map_shape = (self.env.map_size, self.env.map_size, self.index_dim)
        self.buffer_list_shape = (self.agent_num, self.index_dim, self.env.max_buffer_size)
        self.pos_list_shape = (self.agent_num, 2)
        self.bandvec_shape = (self.env.agent_num)
        self.op_shape = (self.index_dim, self.env.max_buffer_size)
        self.move_count, self.move_dict = discrete_circle_sample_count(self.env.move_r)
        self.movemap_shape = (self.env.move_r * 2 + 1, self.env.move_r * 2 + 1)
        self.epsilon = epsilon

        # learning params
        self.tau = tau
        self.cnn_kernel_size = 3
        self.gamma = gamma
        self.lr_aa = lr_aa
        self.lr_ac = lr_ac
        self.lr_ca = lr_ca
        self.lr_cc = lr_cc
        self.batch_size = batch
        self.agent_memory = {}
        self.softmax_memory = {}
        self.center_memory = []
        self.sample_prop = 1 / 4

        # net init
        self.agent_actors = []
        self.center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        self.agent_critics = []
        self.center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)

        self.target_agent_actors = []
        self.target_center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        update_target_net(self.center_actor, self.target_center_actor, tau=0)
        self.target_agent_critics = []
        self.target_center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        update_target_net(self.center_critic, self.target_center_critic, tau=0)

        self.agent_actor_opt = []
        self.agent_critic_opt = []
        self.center_actor_opt = keras.optimizers.Adam(learning_rate=lr_ca)
        self.center_critic_opt = keras.optimizers.Adam(learning_rate=lr_cc)

        self.summaries = {}

        for i in range(self.env.agent_num):
            self.agent_critic_opt.append(keras.optimizers.Adam(learning_rate=lr_ac))
            self.agent_actor_opt.append(keras.optimizers.Adam(learning_rate=lr_aa))
            new_agent_actor = agent_actor([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            target_agent_actor = agent_actor([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            # new_agent_actor = agent_actor([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            # target_agent_actor = agent_actor([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            update_target_net(new_agent_actor, target_agent_actor, tau=0)

            self.agent_actors.append(new_agent_actor)
            self.target_agent_actors.append(target_agent_actor)

            # new_agent_critic = agent_critic([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape,
            #                                  self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            # t_agent_critic = agent_critic([self.state_map_shape, self.pos_shape,
            # self.buffstate_shape, self.buffstate_shape, self.movemap_shape,
            # self.op_shape, self.band_shape], self.cnn_kernel_size)
            new_agent_critic = agent_critic([self.state_map_shape, self.buffstate_shape, self.buffstate_shape,
                                             self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            t_agent_critic = agent_critic([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            update_target_net(new_agent_critic, t_agent_critic, tau=0)
            self.agent_critics.append(new_agent_critic)
            self.target_agent_critics.append(t_agent_critic)

        keras.utils.plot_model(self.center_actor, 'logs/model_figs/new_center_actor.png', show_shapes=True)
        keras.utils.plot_model(self.center_critic, 'logs/model_figs/new_center_critic.png', show_shapes=True)
        keras.utils.plot_model(self.agent_actors[0], 'logs/model_figs/new_agent_actor.png', show_shapes=True)
        keras.utils.plot_model(self.agent_critics[0], 'logs/model_figs/new_agent_critic.png', show_shapes=True)

    def actor_act(self, epoch):
        tmp = random.random()
        if tmp >= self.epsilon and epoch >= 16:
            # agent act
            agent_act_list = []
            softmax_list = []
            cur_state_list = []
            band_vec = np.zeros(self.agent_num)
            for i, agent in enumerate(self.agents):
                # actor = self.agent_actors[i]
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)
                # pos = tf.expand_dims(agent.position, axis=0)
                # print('agent{}pos:{}'.format(i, pos))
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)
                # print('band{}'.format(agent.action.bandwidth))
                band_vec[i] = agent.action.bandwidth
                assemble_state = [state_map, total_data_state, done_data_state, band]
                # print(['agent%s' % i, sum(sum(state_map))])
                cur_state_list.append(assemble_state)
                # print(total_data_state.shape)
                action_output = self.agent_actors[i].predict(assemble_state)
                move_dist = action_output[0][0]
                sio.savemat('debug.mat', {'state': self.env.get_obs(agent), 'move': move_dist})
                # print(move_dist)
                # print(move_dist.shape)
                op_dist = action_output[1][0]
                # print(op_dist.shape)
                # move_ori = np.unravel_index(np.argmax(move_dist), move_dist.shape)
                move_ori = circle_argmax(move_dist, self.env.move_r)
                move = [move_ori[1] - self.env.move_r, move_ori[0] - self.env.move_r]
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.argmax(op_dist[0])] = 1
                offloading[np.argmax(op_dist[1])] = 1
                move_softmax = np.zeros(move_dist.shape)
                op_softmax = np.zeros(self.buffstate_shape)

                move_softmax[move_ori] = 1
                op_softmax[0][np.argmax(op_dist[0])] = 1
                op_softmax[1][np.argmax(op_dist[1])] = 1

                move_softmax = tf.expand_dims(move_softmax, axis=0)
                # move_softmax = tf.expand_dims(move, axis=0)
                op_softmax = tf.expand_dims(op_softmax, axis=0)

                agent_act_list.append([move, execution, offloading])
                softmax_list.append([move_softmax, op_softmax])
            # print(agent_act_list)
            # center act
            done_buffer_list, pos_list = self.env.get_center_state()
            done_buffer_list = tf.expand_dims(done_buffer_list, axis=0)
            # print(done_buffer_list)
            pos_list = tf.expand_dims(pos_list, axis=0)
            band_vec = tf.expand_dims(band_vec, axis=0)
            new_bandvec = self.center_actor.predict([done_buffer_list, pos_list])
            # print('new_bandwidth{}'.format(new_bandvec[0]))

            new_state_maps, new_rewards, done, info = self.env.step(agent_act_list, new_bandvec[0])
            new_done_buffer_list, new_pos_list = self.env.get_center_state()
            new_done_buffer_list = tf.expand_dims(new_done_buffer_list, axis=0)
            new_pos_list = tf.expand_dims(new_pos_list, axis=0)

            # record memory
            for i, agent in enumerate(self.agents):
                state_map = new_state_maps[i]
                # print(['agent%s' % i, sum(sum(state_map))])
                # pos = agent.position
                total_data_state = agent.get_total_data()
                done_data_state = agent.get_done_data()
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)
                # pos = tf.expand_dims(agent.position, axis=0)
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)
                new_states = [state_map, total_data_state, done_data_state, band]
                if agent.no in self.agent_memory.keys():
                    self.agent_memory[agent.no].append([cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]])
                else:
                    self.agent_memory[agent.no] = [[cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]]]

            self.center_memory.append([[done_buffer_list, pos_list], new_bandvec, new_rewards[-1], [new_done_buffer_list, new_pos_list]])

        else:
            # random action
            # agents
            agent_act_list = []
            for i, agent in enumerate(self.agents):
                move = random.sample(list(self.move_dict.values()), 1)[0]
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.random.randint(agent.max_buffer_size)] = 1
                offloading[np.random.randint(agent.max_buffer_size)] = 1
                agent_act_list.append([move, execution, offloading])
            # center
            new_bandvec = np.random.rand(self.agent_num)
            new_bandvec = new_bandvec / np.sum(new_bandvec)
            new_state_maps, new_rewards, done, info = self.env.step(agent_act_list, new_bandvec)

        return new_rewards[-1]

    # @tf.function(experimental_relax_shapes=True)
    def replay(self):
        # agent replay
        for no, agent_memory in self.agent_memory.items():
            if len(agent_memory) < self.batch_size:
                continue
            # print([len(agent_memory[-100:]), self.batch_size])
            samples = agent_memory[-int(self.batch_size * self.sample_prop):] + random.sample(agent_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))
            # t_agent_actor = self.target_agent_actors[no]
            # t_agent_critic = self.target_agent_critics[no]
            # agent_actor = self.agent_actors[no]
            # agent_critic = self.agent_critics[no]
            # state
            state_map = np.vstack([sample[0][0] for sample in samples])
            # pos = np.vstack([sample[0][1] for sample in samples])
            total_data_state = np.vstack([sample[0][1] for sample in samples])
            done_data_state = np.vstack([sample[0][2] for sample in samples])
            band = np.vstack([sample[0][3] for sample in samples])
            # action
            move = np.vstack([sample[1][0] for sample in samples])
            op_softmax = np.vstack([sample[1][1] for sample in samples])
            # reward
            a_reward = tf.expand_dims([sample[2] for sample in samples], axis=-1)
            # new states
            new_state_map = np.vstack([sample[3][0] for sample in samples])
            # new_pos = np.vstack([sample[3][1] for sample in samples])
            new_total_data_state = np.vstack([sample[3][1] for sample in samples])
            new_done_data_state = np.vstack([sample[3][2] for sample in samples])
            new_band = np.vstack([sample[3][3] for sample in samples])
            # # done
            # done = [sample[4] for sample in samples]

            # next actions & rewards
            new_actions = self.target_agent_actors[no].predict([new_state_map, new_total_data_state, new_done_data_state, new_band])
            # new_move = np.array([self.move_dict[np.argmax(single_sample)] for single_sample in new_actions[0]])
            # print(new_actions[1].shape)
            q_future = self.target_agent_critics[no].predict([new_state_map, new_total_data_state, new_done_data_state, new_actions[0], new_actions[1], new_band])
            # print('qfuture{}'.format(q_future))
            target_qs = a_reward + q_future * self.gamma

            # train critic
            with tf.GradientTape() as tape:
                # tape.watch(self.agent_critics[no].trainable_variables)
                q_values = self.agent_critics[no]([state_map, total_data_state, done_data_state, move, op_softmax, band])
                ac_error = q_values - tf.cast(target_qs, dtype=tf.float32)
                # ac_error = q_values - target_qs
                ac_loss = tf.reduce_mean(tf.math.square(ac_error))
            # print('agent%s' % no)
            # print([q_values, target_qs, ac_error, ac_loss])
            ac_grad = tape.gradient(ac_loss, self.agent_critics[no].trainable_variables)
            # print(ac_grad)
            self.agent_critic_opt[no].apply_gradients(zip(ac_grad, self.agent_critics[no].trainable_variables))

            # train actor
            with tf.GradientTape() as tape:
                tape.watch(self.agent_actors[no].trainable_variables)
                actions = self.agent_actors[no]([state_map, total_data_state, done_data_state, band])
                # actor_move = np.array([self.move_dict[np.argmax(single_sample)] for single_sample in actions[0]])
                new_r = self.agent_critics[no]([state_map, total_data_state, done_data_state, actions[0], actions[1], band])
                # print(new_r)
                aa_loss = tf.reduce_mean(new_r)
                # print(aa_loss)
            aa_grad = tape.gradient(aa_loss, self.agent_actors[no].trainable_variables)
            # print(aa_grad)
            self.agent_actor_opt[no].apply_gradients(zip(aa_grad, self.agent_actors[no].trainable_variables))

            # summary info
            self.summaries['agent%s-critic_loss' % no] = ac_loss
            self.summaries['agent%s-actor_loss' % no] = aa_loss

        # center replay
        if len(self.center_memory) < self.batch_size:
            return
        center_samples = self.center_memory[-int(self.batch_size * self.sample_prop):] + random.sample(self.center_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))
        done_buffer_list = np.vstack([sample[0][0] for sample in center_samples])
        pos_list = np.vstack([sample[0][1] for sample in center_samples])
        bandvec_act = np.vstack([sample[1] for sample in center_samples])
        c_reward = tf.expand_dims([sample[2] for sample in center_samples], axis=-1)
        # new states
        new_done_buffer_list = np.vstack([sample[3][0] for sample in center_samples])
        new_pos_list = np.vstack([sample[3][1] for sample in center_samples])
        # next actions & reward
        new_c_actions = self.target_center_actor.predict([new_done_buffer_list, new_pos_list])
        cq_future = self.target_center_critic.predict([new_done_buffer_list, new_pos_list, new_c_actions])
        c_target_qs = c_reward + cq_future * self.gamma
        self.summaries['cq_val'] = np.average(c_reward[0])

        # train center critic
        with tf.GradientTape() as tape:
            tape.watch(self.center_critic.trainable_variables)
            cq_values = self.center_critic([done_buffer_list, pos_list, bandvec_act])
            cc_loss = tf.reduce_mean(tf.math.square(cq_values - tf.cast(c_target_qs, dtype=tf.float32)))
            # cc_loss = tf.reduce_mean(tf.math.square(cq_values - c_target_qs))
        cc_grad = tape.gradient(cc_loss, self.center_critic.trainable_variables)
        self.center_critic_opt.apply_gradients(zip(cc_grad, self.center_critic.trainable_variables))
        # train center actor
        with tf.GradientTape() as tape:
            tape.watch(self.center_actor.trainable_variables)
            c_act = self.center_actor([done_buffer_list, pos_list])
            ca_loss = tf.reduce_mean(self.center_critic([done_buffer_list, pos_list, c_act]))
        # print(self.center_critic([sensor_maps, agent_maps, c_act]))
        ca_grad = tape.gradient(ca_loss, self.center_actor.trainable_variables)
        # print(ca_grad)
        self.center_actor_opt.apply_gradients(zip(ca_grad, self.center_actor.trainable_variables))
        # print(ca_loss)
        self.summaries['center-critic_loss'] = cc_loss
        self.summaries['center-actor_loss'] = ca_loss

    def save_model(self, episode, time_str):
        for i in range(self.agent_num):
            self.agent_actors[i].save('logs/models/{}/agent-actor-{}_episode{}.h5'.format(time_str, i, episode))
            self.agent_critics[i].save('logs/models/{}/agent-critic-{}_episode{}.h5'.format(time_str, i, episode))
        self.center_actor.save('logs/models/{}/center-actor_episode{}.h5'.format(time_str, episode))
        self.center_critic.save('logs/models/{}/center-critic_episode{}.h5'.format(time_str, episode))

    # @tf.function
    def train(self, max_epochs=2000, max_step=500, up_freq=8, render=False, render_freq=1, FL=False, FL_omega=0.5, anomaly_edge=False):
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = 'logs/fit/' + cur_time
        env_log_dir = 'logs/env/env' + cur_time
        record_dir = 'logs/records/' + cur_time
        os.mkdir(env_log_dir)
        os.mkdir(record_dir)
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        # tf.summary.trace_on(graph=True, profiler=True)
        os.makedirs('logs/models/' + cur_time)
        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        finish_length = []
        finish_size = []
        sensor_ages = []
        # sensor_map = self.env.DS_map
        # sensor_pos_list = self.env.world.sensor_pos
        # sensor_states = [self.env.DS_state]
        # agent_pos = [[[agent.position[0], agent.position[1]] for agent in self.agents]]
        # agent_off = [[agent.action.offloading for agent in self.agents]]
        # agent_exe = [[agent.action.execution for agent in self.agents]]
        # agent_band = [[agent.action.bandwidth for agent in self.agents]]
        # agent_trans = [[agent.trans_rate for agent in self.agents]]
        # buff, pos = self.env.get_center_state()
        # agent_donebuff = [buff]
        # exe, done = self.env.get_buffer_state()
        # exebuff = [exe]
        # donebuff = [done]

        anomaly_step = 6000
        anomaly_agent = self.agent_num - 1

        # if anomaly_edge:
        #     anomaly_step = np.random.randint(int(max_epochs * 0.5), int(max_epochs * 0.75))
        #     anomaly_agent = np.random.randint(self.agent_num)
        # summary_record = []

        while epoch < max_epochs:
            print('epoch%s' % epoch)
            # if anomaly_edge and (epoch == anomaly_step):
            #     self.agents[anomaly_agent].movable = False

            if render and (epoch % 20 == 1):
                self.env.render(env_log_dir, epoch, True)
                # sensor_states.append(self.env.DS_state)

            if steps >= max_step:
                # self.env.world.finished_data = []
                episode += 1
                # self.env.reset()
                for m in self.agent_memory.keys():
                    del self.agent_memory[m][0:-self.batch_size * 2]
                del self.center_memory[0:-self.batch_size * 2]
                print('episode {}: {} total reward, {} steps, {} epochs'.format(episode, total_reward / steps, steps, epoch))

                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)
                    # tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=train_log_dir)

                summary_writer.flush()
                self.save_model(episode, cur_time)
                steps = 0
                total_reward = 0

            cur_reward = self.actor_act(epoch)
            # print('episode-%s reward:%f' % (episode, cur_reward))
            self.replay()
            finish_length.append(len(self.env.world.finished_data))
            finish_size.append(sum([data[0] for data in self.env.world.finished_data]))
            sensor_ages.append(list(self.env.world.sensor_age.values()))
            # agent_pos.append([[agent.position[0], agent.position[1]] for agent in self.env.world.agents])
            # # print(agent_pos)
            # agent_off.append([agent.action.offloading for agent in self.agents])
            # agent_exe.append([agent.action.execution for agent in self.agents])
            # # agent_band.append([agent.action.bandwidth for agent in self.agents])
            # agent_trans.append([agent.trans_rate for agent in self.agents])
            # buff, pos = self.env.get_center_state()
            # # agent_donebuff.append(buff)
            # exe, done = self.env.get_buffer_state()
            # exebuff.append(exe)
            # donebuff.append(done)

            # summary_record.append(self.summaries)
            # update target
            if epoch % up_freq == 1:
                print('update targets, finished data: {}'.format(len(self.env.world.finished_data)))

               # finish_length.append(len(self.env.world.finished_data))
                if FL:
                    merge_fl(self.agent_actors, FL_omega)
                    merge_fl(self.agent_critics, FL_omega)
                    # merge_fl(self.target_agent_actors, FL_omega)
                    # merge_fl(self.target_agent_critics, FL_omega)
                for i in range(self.agent_num):
                    update_target_net(self.agent_actors[i], self.target_agent_actors[i], self.tau)
                    update_target_net(self.agent_critics[i], self.target_agent_critics[i], self.tau)
                update_target_net(self.center_actor, self.target_center_actor, self.tau)
                update_target_net(self.center_critic, self.target_center_critic, self.tau)

            total_reward += cur_reward
            steps += 1
            epoch += 1

            # tensorboard
            with summary_writer.as_default():
                if len(self.center_memory) > self.batch_size:
                    tf.summary.scalar('Loss/center_actor_loss', self.summaries['center-actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/center_critic_loss', self.summaries['center-critic_loss'], step=epoch)
                    tf.summary.scalar('Loss/agent_actor_loss', self.summaries['agent0-actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/agent_critic_loss', self.summaries['agent0-critic_loss'], step=epoch)
                    tf.summary.scalar('Stats/cq_val', self.summaries['cq_val'], step=epoch)
                    for acount in range(self.agent_num):
                        tf.summary.scalar('Stats/agent%s_actor_loss' % acount, self.summaries['agent%s-actor_loss' % acount], step=epoch)
                        tf.summary.scalar('Stats/agent%s_critic_loss' % acount, self.summaries['agent%s-critic_loss' % acount], step=epoch)
                tf.summary.scalar('Main/step_average_age', cur_reward, step=epoch)

            summary_writer.flush()

        # save final model
        self.save_model(episode, cur_time)
        sio.savemat(record_dir + '/data.mat',
                    {'finish_len': finish_length,
                     'finish_data': finish_size,
                     'ages': sensor_ages})
        # sio.savemat(record_dir + '/data.mat',
        #             {'finish_len': finish_length,
        #              'finish_data': finish_size,
        #              'sensor_map': sensor_map,
        #              'sensor_list': sensor_pos_list,
        #              'sensor_state': sensor_states,
        #              'agentpos': agent_pos,
        #              'agentoff': agent_off,
        #              'agentexe': agent_exe,
        #              'agenttran': agent_trans,
        #              'agentbuff': agent_donebuff,
        #              'agentexebuff': exebuff,
        #              'agentdonebuff': donebuff,
        #              'agentband': agent_band,
        #              'anomaly': [anomaly_step,
        #                          anomaly_agent]})
        # with open(record_dir + '/record.json', 'w') as f:
        #     json.dump(summary_record, f)

        # gif
        self.env.render(env_log_dir, epoch, True)
        img_paths = glob.glob(env_log_dir + '/*.png')
        img_paths.sort(key=lambda x: int(x.split('.')[0].split('\\')[-1]))

        gif_images = []
        for path in img_paths:
            gif_images.append(imageio.imread(path))
        imageio.mimsave(env_log_dir + '/all.gif', gif_images, fps=15)
