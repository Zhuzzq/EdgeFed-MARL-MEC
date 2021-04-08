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


def get_center_state(env):
    total_buffer_list = np.zeros([env.agent_num, 2, env.max_buffer_size])
    done_buffer_list = np.zeros([env.agent_num, 2, env.max_buffer_size])
    pos_list = np.zeros([env.agent_num, 2])
    for i, agent in enumerate(env.agents):
        pos_list[i] = agent.position
        for j, d in enumerate(list(agent.total_data.values())):
            total_buffer_list[i][0][j] = d[0]
            total_buffer_list[i][1][j] = d[1]
        for j, d in enumerate(agent.done_data):
            done_buffer_list[i][0][j] = d[0]
            done_buffer_list[i][1][j] = d[1]
    # print(buffer_list)
    # print(pos_list)
    return total_buffer_list, done_buffer_list, pos_list


def discrete_circle_sample_count(n):
    count = 0
    move_dict = {}
    for x in range(-n, n + 1):
        y_l = int(np.floor(np.sqrt(n**2 - x**2)))
        for y in range(-y_l, y_l + 1):
            move_dict[count] = np.array([y, x])
            count += 1
    return (count), move_dict


def center_actor(input_dim_list, cnn_kernel_size, move_r, kernel_num):
    sensor_map = keras.Input(shape=input_dim_list[0])
    total_buffer_list = keras.Input(shape=input_dim_list[1])
    done_buffer_list = keras.Input(shape=input_dim_list[2])
    pos_list = keras.Input(shape=input_dim_list[3])

    # CNN for map
    cnn_map = layers.Dense(1, activation='relu')(sensor_map)
    cnn_map = layers.Conv2D(filters=kernel_num, kernel_size=cnn_kernel_size, activation='relu', padding='same')(cnn_map)
    cnn_map = layers.AveragePooling2D(pool_size=int(input_dim_list[0][0] / (2 * move_r + 1)))(cnn_map)
    cnn_map = layers.AlphaDropout(0.2)(cnn_map)
    move_out = tf.transpose(cnn_map, perm=[0, 3, 1, 2])
    move_out = tf.expand_dims(move_out, axis=-1)

    # buffer
    total_buffer = tf.transpose(total_buffer_list, perm=[0, 1, 3, 2])
    total_buffer = layers.Dense(1, activation='relu')(total_buffer)
    total_buffer = tf.squeeze(total_buffer, axis=-1)
    exe_op = layers.Dense(input_dim_list[1][2], activation='softmax')(total_buffer)

    done_buffer = tf.transpose(done_buffer_list, perm=[0, 1, 3, 2])
    done_buffer = layers.Dense(1, activation='relu')(done_buffer)
    done_buffer = tf.squeeze(done_buffer, axis=-1)
    off_op = layers.Dense(input_dim_list[2][2], activation='softmax')(done_buffer)

    # center
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)
    buffer_state = tf.squeeze(buffer_state, axis=-1)
    # pos list
    pos = layers.Dense(2, activation='relu')(pos_list)

    bandwidth_out = layers.concatenate([buffer_state, pos], axis=-1)
    # bandwidth_out = layers.AlphaDropout(0.2)(bandwidth_out)
    bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)
    bandwidth_out = tf.squeeze(bandwidth_out, axis=-1)
    # bandwidth_out = layers.Dense(input_dim_list[2], activation='relu')(bandwidth_out)
    bandwidth_out = layers.Softmax()(bandwidth_out)
    # bandwidth_out += 1 / (input_dim_list[3][0] * 5)
    # bandwidth_out = bandwidth_out / tf.reduce_sum(bandwidth_out, 1, keepdims=True)
    # bandwidth_out = bandwidth_out / tf.expand_dims(tf.reduce_sum(bandwidth_out, 1), axis=-1)

    model = keras.Model(inputs=[sensor_map, total_buffer_list, done_buffer_list, pos_list], outputs=[move_out, exe_op, off_op, bandwidth_out], name='center_actor_net')
    return model


# center critic
def center_critic(input_dim_list, cnn_kernel_size):
    sensor_map = keras.Input(shape=input_dim_list[0])
    total_buffer_list = keras.Input(shape=input_dim_list[1])
    done_buffer_list = keras.Input(shape=input_dim_list[2])
    pos_list = keras.Input(shape=input_dim_list[3])
    move = keras.Input(shape=input_dim_list[4])
    exe_op = keras.Input(shape=input_dim_list[5])
    off_op = keras.Input(shape=input_dim_list[6])
    bandwidth_vec = keras.Input(shape=input_dim_list[7])

    # map
    cnn_map = layers.Dense(1, activation='relu')(sensor_map)
    cnn_map = layers.Conv2D(filters=1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(cnn_map)
    cnn_map = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(cnn_map)
    cnn_map = layers.AlphaDropout(0.2)(cnn_map)
    cnn_map = layers.Flatten()(cnn_map)
    cnn_map = layers.Dense(2, activation='relu')(cnn_map)

    # buffer
    total_buffer_state = layers.Dense(1, activation='relu')(total_buffer_list)
    total_buffer_state = tf.squeeze(total_buffer_state, axis=-1)
    total_buffer_state = layers.Dense(1, activation='relu')(total_buffer_state)
    total_buffer_state = tf.squeeze(total_buffer_state, axis=-1)

    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)
    buffer_state = tf.squeeze(buffer_state, axis=-1)
    buffer_state = layers.Dense(1, activation='relu')(buffer_state)
    buffer_state = tf.squeeze(buffer_state, axis=-1)

    # pos list
    pos = layers.Dense(1, activation='relu')(pos_list)
    pos = tf.squeeze(pos, axis=-1)

    move_mlp = layers.Flatten()(move)
    move_mlp = layers.Dense(1, activation='relu')(move_mlp)

    exe_mlp = layers.Flatten()(exe_op)
    exe_mlp = layers.Dense(1, activation='relu')(exe_mlp)

    off_mlp = layers.Flatten()(off_op)
    off_mlp = layers.Dense(1, activation='relu')(off_mlp)
    # bandvec
    # band_in = layers.Dense(2, activation='relu')(bandwidth_vec)

    r_out = layers.concatenate([cnn_map, total_buffer_state, buffer_state, pos, move_mlp, exe_mlp, off_mlp, bandwidth_vec])
    # r_out = layers.AlphaDropout(0.2)(r_out)
    r_out = layers.Dense(1, activation='relu')(r_out)
    model = keras.Model(inputs=[sensor_map, total_buffer_list, done_buffer_list, pos_list, move, exe_op, off_op, bandwidth_vec], outputs=r_out, name='center_critic_net')
    return model


def update_target_net(model, target, tau=0.8):
    weights = model.get_weights()
    target_weights = target.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * (1 - tau) + target_weights[i] * tau
    target.set_weights(target_weights)


def circle_argmax(move_dist, move_r):
    max_pos = np.argwhere(tf.squeeze(move_dist, axis=-1) == np.max(move_dist))
    # print(tf.squeeze(move_dist, axis=-1))
    pos_dist = np.linalg.norm(max_pos - np.array([move_r, move_r]), axis=1)
    # print(max_pos)
    return max_pos[np.argmin(pos_dist)]


class ACAgent(object):
    def __init__(self, env, tau, gamma, lr_aa, lr_ac, lr_ca, lr_cc, batch, epsilon=0.2):
        self.env = env
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.index_dim = 2
        self.obs_r = self.env.obs_r
        self.state_map_shape = (self.obs_r * 2 + 1, self.obs_r * 2 + 1, self.index_dim)
        self.pos_shape = (2)
        self.band_shape = (1)
        self.sensor_map_shape = (self.env.map_size, self.env.map_size, self.index_dim)
        self.buffer_list_shape = (self.agent_num, self.index_dim, self.env.max_buffer_size)
        self.pos_list_shape = (self.agent_num, 2)
        self.bandvec_shape = (self.env.agent_num)
        self.op_shape = (self.agent_num, self.env.max_buffer_size)
        self.move_count, self.move_dict = discrete_circle_sample_count(self.env.move_r)
        self.movemap_shape = (self.agent_num, self.env.move_r * 2 + 1, self.env.move_r * 2 + 1)
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
        self.softmax_memory = {}
        self.center_memory = []
        self.sample_prop = 1 / 4

        # net init
        self.center_actor = center_actor([self.sensor_map_shape, self.buffer_list_shape, self.buffer_list_shape, self.pos_list_shape], self.cnn_kernel_size, self.env.move_r, self.agent_num)
        self.center_critic = center_critic([self.sensor_map_shape, self.buffer_list_shape, self.buffer_list_shape, self.pos_list_shape,
                                            self.movemap_shape, self.op_shape, self.op_shape, self.bandvec_shape], self.cnn_kernel_size)

        self.target_center_actor = center_actor([self.sensor_map_shape, self.buffer_list_shape, self.buffer_list_shape, self.pos_list_shape], self.cnn_kernel_size, self.env.move_r, self.agent_num)

        update_target_net(self.center_actor, self.target_center_actor, tau=0)
        self.target_center_critic = center_critic([self.sensor_map_shape, self.buffer_list_shape, self.buffer_list_shape, self.pos_list_shape,
                                                   self.movemap_shape, self.op_shape, self.op_shape, self.bandvec_shape], self.cnn_kernel_size)
        update_target_net(self.center_critic, self.target_center_critic, tau=0)

        self.center_actor_opt = keras.optimizers.Adam(learning_rate=lr_ca)
        self.center_critic_opt = keras.optimizers.Adam(learning_rate=lr_cc)

        self.summaries = {}

        keras.utils.plot_model(self.center_actor, 'logs/model_figs/baseline_actor.png', show_shapes=True)
        keras.utils.plot_model(self.center_critic, 'logs/model_figs/baseline_critic.png', show_shapes=True)

    def actor_act(self, epoch):
        tmp = random.random()
        if tmp >= self.epsilon and epoch >= 16:
            # agent act
            agent_act_list = []
            softmax_list = []
            cur_state_list = []
            band_vec = np.zeros(self.agent_num)

            # print(agent_act_list)
            # center act
            sensor_map, agent_map = self.env.get_statemap()
            total_buffer_list, done_buffer_list, pos_list = get_center_state(self.env)
            sensor_map = tf.expand_dims(sensor_map, axis=0)
            total_buffer_list = tf.expand_dims(total_buffer_list, axis=0)
            done_buffer_list = tf.expand_dims(done_buffer_list, axis=0)
            # print(done_buffer_list)
            pos_list = tf.expand_dims(pos_list, axis=0)
            band_vec = tf.expand_dims(band_vec, axis=0)
            # print([sensor_map.shape, total_buffer_list.shape, done_buffer_list.shape, pos_list.shape])
            action = self.center_actor.predict([sensor_map, total_buffer_list, done_buffer_list, pos_list])
            new_bandvec = action[3][0]
            # print('new_bandwidth{}'.format(new_bandvec[0]))
            for i, agent in enumerate(self.agents):
                move_dist = action[0][0][i]

                # print(move_dist)
                # print(move_dist.shape)
                exe_dist = action[1][0][i]
                off_dist = action[2][0][i]
                # print(op_dist.shape)
                # move_ori = np.unravel_index(np.argmax(move_dist), move_dist.shape)
                move_ori = circle_argmax(move_dist, self.env.move_r)
                move = [move_ori[1] - self.env.move_r, move_ori[0] - self.env.move_r]
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.argmax(exe_dist)] = 1
                offloading[np.argmax(off_dist)] = 1
                move_softmax = np.zeros(move_dist.shape)

                move_softmax[move_ori] = 1

                move_softmax = tf.expand_dims(move_softmax, axis=0)
                # move_softmax = tf.expand_dims(move, axis=0)
                agent_act_list.append([move, execution, offloading])

            new_state_map, new_rewards, done, info = self.env.step(agent_act_list, new_bandvec)
            new_sensor_map, agent_map = self.env.get_statemap()
            new_total_buffer_list, new_done_buffer_list, new_pos_list = get_center_state(self.env)
            new_total_buffer_list = tf.expand_dims(new_total_buffer_list, axis=0)
            new_done_buffer_list = tf.expand_dims(new_done_buffer_list, axis=0)
            new_pos_list = tf.expand_dims(new_pos_list, axis=0)

            # record memory
            self.center_memory.append([[sensor_map, total_buffer_list, done_buffer_list, pos_list], action, new_rewards[-1],
                                       [new_sensor_map, new_total_buffer_list, new_done_buffer_list, new_pos_list]])

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
        # center replay
        if len(self.center_memory) < self.batch_size:
            return
        center_samples = self.center_memory[-int(self.batch_size * self.sample_prop):] + random.sample(self.center_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))
        sensor_map = np.vstack([sample[0][0] for sample in center_samples])
        total_buffer_list = np.vstack([sample[0][1] for sample in center_samples])
        done_buffer_list = np.vstack([sample[0][2] for sample in center_samples])
        pos_list = np.vstack([sample[0][3] for sample in center_samples])
        # print(center_samples[0][1])
        # act = np.vstack([sample[1] for sample in center_samples])
        move = np.vstack([sample[1][0] for sample in center_samples])
        exe = np.vstack([sample[1][1] for sample in center_samples])
        off = np.vstack([sample[1][2] for sample in center_samples])
        band_act = np.vstack([sample[1][3] for sample in center_samples])
        c_reward = tf.expand_dims([sample[2] for sample in center_samples], axis=-1)
        # new states
        new_sensor_map = np.stack([sample[3][0] for sample in center_samples], axis=0)
        new_total_buffer_list = np.vstack([sample[3][1] for sample in center_samples])
        new_done_buffer_list = np.vstack([sample[3][2] for sample in center_samples])
        new_pos_list = np.vstack([sample[3][3] for sample in center_samples])
        # next actions & reward
        new_c_actions = self.target_center_actor.predict([new_sensor_map, new_total_buffer_list, new_done_buffer_list, new_pos_list])
        cq_future = self.target_center_critic.predict([new_sensor_map, new_total_buffer_list, new_done_buffer_list, new_pos_list,
                                                       new_c_actions[0], new_c_actions[1], new_c_actions[2], new_c_actions[3]])
        c_target_qs = c_reward + cq_future * self.gamma
        self.summaries['cq_val'] = np.average(c_reward[0])

        # train center critic
        with tf.GradientTape() as tape:
            tape.watch(self.center_critic.trainable_variables)
            cq_values = self.center_critic([sensor_map, total_buffer_list, done_buffer_list, pos_list, move, exe, off, band_act])
            cc_loss = tf.reduce_mean(tf.math.square(cq_values - tf.cast(c_target_qs, dtype=tf.float32)))
            # cc_loss = tf.reduce_mean(tf.math.square(cq_values - c_target_qs))
        cc_grad = tape.gradient(cc_loss, self.center_critic.trainable_variables)
        self.center_critic_opt.apply_gradients(zip(cc_grad, self.center_critic.trainable_variables))
        # train center actor
        with tf.GradientTape() as tape:
            tape.watch(self.center_actor.trainable_variables)
            c_act = self.center_actor([sensor_map, total_buffer_list, done_buffer_list, pos_list])
            ca_loss = tf.reduce_mean(self.center_critic([sensor_map, total_buffer_list, done_buffer_list, pos_list, c_act[0], c_act[1], c_act[2], c_act[3]]))
        # print(self.center_critic([sensor_maps, agent_maps, c_act]))
        ca_grad = tape.gradient(ca_loss, self.center_actor.trainable_variables)
        # print(ca_grad)
        self.center_actor_opt.apply_gradients(zip(ca_grad, self.center_actor.trainable_variables))
        # print(ca_loss)
        self.summaries['center-critic_loss'] = cc_loss
        self.summaries['center-actor_loss'] = ca_loss

    def save_model(self, episode, time_str):
        self.center_actor.save('logs/models/{}/center-actor_episode{}.h5'.format(time_str, episode))
        self.center_critic.save('logs/models/{}/center-critic_episode{}.h5'.format(time_str, episode))

    # @tf.function
    def train(self, max_epochs=2000, max_step=500, up_freq=8, render=False, render_freq=1):
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
        # summary_record = []

        while epoch < max_epochs:
            print('epoch%s' % epoch)
            if render and (epoch % 32 == 1):
                self.env.render(env_log_dir, epoch, True)

            if steps >= max_step:
                # self.env.world.finished_data = []
                episode += 1
                # self.env.reset()
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
            # summary_record.append(self.summaries)

            # update target
            if epoch % up_freq == 1:
                print('update targets, finished data: {}'.format(len(self.env.world.finished_data)))
                # finish_length.append(len(self.env.world.finished_data))
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
                    tf.summary.scalar('Stats/cq_val', self.summaries['cq_val'], step=epoch)
                tf.summary.scalar('Main/step_average_age', cur_reward, step=epoch)

            summary_writer.flush()

        # save final model
        self.save_model(episode, cur_time)
        sio.savemat(record_dir + '/data.mat', {'finish_len': finish_length, 'finish_data': finish_size, 'ages': sensor_ages})
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
