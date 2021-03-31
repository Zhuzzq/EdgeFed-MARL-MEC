# -*- coding: UTF-8 -*-
import numpy as np
import gym
from gym import spaces
import numpy as np
from .space_def import circle_space
from .space_def import onehot_space
from .space_def import sum_space
from gym.envs.registration import EnvSpec
import logging
from matplotlib import pyplot as plt
from IPython import display

logging.basicConfig(level=logging.WARNING)

# plt.figure()
# plt.ion()


def get_circle_plot(pos, r):
    x_c = np.arange(-r, r, 0.01)
    up_y = np.sqrt(r**2 - np.square(x_c))
    down_y = - up_y
    x = x_c + pos[0]
    y1 = up_y + pos[1]
    y2 = down_y + pos[1]
    return [x, y1, y2]


class MEC_MARL_ENV(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, world, alpha=0.5, beta=0.2, aggregate_reward=False, discrete=True,
                 reset_callback=None, info_callback=None, done_callback=None):
        # system initialize
        self.world = world
        self.obs_r = world.obs_r
        self.move_r = world.move_r
        self.collect_r = world.collect_r
        self.max_buffer_size = self.world.max_buffer_size
        self.agents = self.world.agents
        self.agent_num = self.world.agent_count
        self.sensor_num = self.world.sensor_count
        self.sensors = self.world.sensors
        self.DS_map = self.world.DS_map
        self.map_size = self.world.map_size
        self.DS_state = self.world.DS_state
        self.alpha = alpha
        self.beta = beta

        self.reset_callback = reset_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # game mode
        self.aggregate_reward = aggregate_reward  # share same rewards
        self.discrete_flag = discrete
        self.state = None
        self.time = 0
        self.images = []

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            if self.discrete_flag:
                act_space = spaces.Tuple((circle_space.Discrete_Circle(
                    agent.move_r), onehot_space.OneHot(self.max_buffer_size), sum_space.SumOne(self.agent_num), onehot_space.OneHot(self.max_buffer_size)))
                # move, offloading(boolxn), bandwidth([0,1]), execution
                obs_space = spaces.Tuple((spaces.MultiDiscrete(
                    [self.map_size, self.map_size]), spaces.Box(0, np.inf, [agent.obs_r * 2, agent.obs_r * 2, 2])))
                # pos, obs map
                self.action_space.append(act_space)
                self.observation_space.append(obs_space)
        self.render()

    def step(self, agent_action, center_action):
        obs = []
        reward = []
        done = []
        info = {'n': []}
        self.agents = self.world.agents

        # world step
        logging.info("set actions")
        for i, agent in enumerate(self.agents):
            self._set_action(agent_action[i], center_action, agent)

        # world update
        self.world.step()
        # new observation
        logging.info("agent observation")
        for agent in self.agents:
            obs.append(self.get_obs(agent))
            done.append(self._get_done(agent))
            reward.append(self._get_age())  # to do
            info['n'].append(self._get_info(agent))
        self.state = obs
        # reward
        reward_sum = np.sum(reward)
        logging.info("get reward")
        if self.aggregate_reward:
            reward = [reward_sum] * self.agent_num
        return self.state, reward, done, info

    def reset(self):
        # reset world
        self.world.finished_data = []
        # reset renderer
        # self._reset_render()
        # record observations for each agent
        for sensor in self.sensors:
            sensor.data_buffer = []
            sensor.collect_state = False
        for agent in self.agents:
            agent.idle = True
            agent.data_buffer = {}
            agent.total_data = {}
            agent.done_data = []
            agent.collecting_sensors = []

    def _set_action(self, act, center_action, agent):
        agent.action.move = np.zeros(2)
        agent.action.execution = act[1]
        agent.action.bandwidth = center_action[agent.no]
        if agent.movable and agent.idle:
            # print([agent.no, act[0]])
            if np.linalg.norm(act[0]) > agent.move_r:
                act[0] = [int(act[0][0] * agent.move_r / np.linalg.norm(act[0])), int(act[0][1] * agent.move_r / np.linalg.norm(act[0]))]
            if not np.count_nonzero(act[0]) and np.random.rand() > 0.5:
                mod_x = np.random.normal(loc=0, scale=1)
                mod_y = np.random.normal(loc=0, scale=1)
                mod_x = int(min(max(-1, mod_x), 1) * agent.move_r / 2)
                mod_y = int(min(max(-1, mod_y), 1) * agent.move_r / 2)
                act[0] = [mod_x, mod_y]
            agent.action.move = np.array(act[0])
            new_x = agent.position[0] + agent.action.move[0]
            new_y = agent.position[1] + agent.action.move[1]
            if new_x < 0 or new_x > self.map_size - 1:
                agent.action.move[0] = -agent.action.move[0]
            if new_y < 0 or new_y > self.map_size - 1:
                agent.action.move[1] = -agent.action.move[1]
            agent.position += agent.action.move
            # agent.position = np.array([max(0, agent.position[0]),
            #                            max(0, agent.position[1])])
            # agent.position = np.array([min(self.map_size - 1, agent.position[0]), min(
            #     self.map_size - 1, agent.position[1])])
        if agent.offloading_idle:
            agent.action.offloading = act[2]
        print('agent-{} action: move{}, exe{},off{},band{}'.format(agent.no, agent.action.move, agent.action.execution, agent.action.offloading, agent.action.bandwidth))

    # get info used for benchmarking

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def get_obs(self, agent):
        obs = np.zeros([agent.obs_r * 2 + 1, agent.obs_r * 2 + 1, 2])
        # left up point
        lu = [max(0, agent.position[0] - agent.obs_r),
              min(self.map_size, agent.position[1] + agent.obs_r + 1)]
        # right down point
        rd = [min(self.map_size, agent.position[0] + agent.obs_r + 1),
              max(0, agent.position[1] - agent.obs_r)]

        # ob_map position
        ob_lu = [agent.obs_r - agent.position[0] + lu[0],
                 agent.obs_r - agent.position[1] + lu[1]]
        ob_rd = [agent.obs_r + rd[0] - agent.position[0],
                 agent.obs_r + rd[1] - agent.position[1]]
        # print([lu, rd, ob_lu, ob_rd])
        for i in range(ob_rd[1], ob_lu[1]):
            map_i = rd[1] + i - ob_rd[1]
            # print([i, map_i])
            obs[i][ob_lu[0]:ob_rd[0]] = self.DS_state[map_i][lu[0]:rd[0]]
        # print(self.DS_state[ob_rd[1]][ob_lu[0]:ob_rd[0]].shape)
        agent.obs = obs
        # print(obs.shape)
        return obs

    def get_statemap(self):
        sensor_map = np.ones([self.map_size, self.map_size, 2])
        agent_map = np.ones([self.map_size, self.map_size, 2])
        for sensor in self.sensors:
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][0] = sum([i[0] for i in sensor.data_buffer])
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][1] = sum([i[1] for i in sensor.data_buffer]) / max(len(sensor.data_buffer), 1)
        for agent in self.agents:
            agent_map[int(agent.position[1])][int(agent.position[0])][0] = sum([i[0] for i in agent.done_data])
            agent_map[int(agent.position[1])][int(agent.position[0])][1] = sum([i[1] for i in agent.done_data]) / max(len(agent.done_data), 1)
        return sensor_map, agent_map
        # get dones for a particular agent
        # unused right now -- agents are allowed to go beyond the viewing screen

    def get_center_state(self):
        buffer_list = np.zeros([self.agent_num, 2, self.max_buffer_size])
        pos_list = np.zeros([self.agent_num, 2])
        for i, agent in enumerate(self.agents):
            pos_list[i] = agent.position
            for j, d in enumerate(agent.done_data):
                buffer_list[i][0][j] = d[0]
                buffer_list[i][1][j] = d[1]
        # print(buffer_list)
        # print(pos_list)
        return buffer_list, pos_list

    def get_buffer_state(self):
        exe = []
        done = []
        for agent in self.agents:
            exe.append(len(agent.total_data))
            done.append(len(agent.done_data))
        return exe, done

    def _get_done(self, agent):
        if self.done_callback is None:
            return 0
        return self.done_callback(agent, self.world)

    # average age
    def _get_age(self):
        return np.mean(list(self.world.sensor_age.values()))

    # get reward for a particular agent
    def _get_reward(self):
        return np.mean(list(self.world.sensor_age.values()))
        # state_reward = sum(sum(self.DS_state)) / self.sensor_num
        # done_reward = [[i[0], i[1]] for i in self.world.finished_data]
        # if not done_reward:
        #     done_reward = np.array([0, 0])
        # else:
        #     # print(np.array(done_reward))
        #     done_reward = np.average(np.array(done_reward), axis=0)
        # buffer_reward = 0
        # for agent in self.agents:
        #     if agent.done_data:
        #         buffer_reward += np.mean([d[1] for d in agent.done_data])
        # buffer_reward = buffer_reward / self.agent_num
        # # print(buffer_reward)
        # # print([state_reward, done_reward])
        # return self.alpha * done_reward[1] + self.beta * (state_reward[1] + self.sensor_num - self.map_size * self.map_size) + (1 - self.alpha - self.beta) * buffer_reward

    def render(self, name=None, epoch=None, save=False):
        # plt.subplot(1,3,1)
        # plt.scatter(self.world.sensor_pos[0],self.world.sensor_pos[1],alpha=0.7)
        # plt.grid()
        # plt.title('sensor position')
        # plt.subplot(1,3,2)
        # plt.scatter(self.world.agent_pos_init[0],self.world.agent_pos_init[1],alpha=0.7)
        # plt.grid()
        # plt.title('agent initial position')
        # plt.subplot(1,3,3)
        plt.figure()
        plt.scatter(self.world.sensor_pos[0], self.world.sensor_pos[1], c='cornflowerblue', alpha=0.9)

        for agent in self.world.agents:
            plt.scatter(agent.position[0], agent.position[1], c='orangered', alpha=0.9)
            plt.annotate(agent.no + 1, xy=(agent.position[0], agent.position[1]), xytext=(agent.position[0] + 0.1, agent.position[1] + 0.1))
            obs_plot = get_circle_plot(agent.position, self.obs_r)
            collect_plot = get_circle_plot(agent.position, self.collect_r)
            plt.fill_between(obs_plot[0], obs_plot[1], obs_plot[2], where=obs_plot[1] > obs_plot[2], color='darkorange', alpha=0.02)
            plt.fill_between(collect_plot[0], collect_plot[1], collect_plot[2], where=collect_plot[1] > collect_plot[2], color='darkorange', alpha=0.05)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Sensors', 'Edge Agents'])
        plt.axis('square')
        plt.xlim([0, self.map_size])
        plt.ylim([0, self.map_size])
        plt.title('all entity position(epoch%s)' % epoch)
        if not save:
            plt.show()
            return
        plt.savefig('%s/%s.png' % (name, epoch))
        plt.close()
        # plt.pause(0.3)
        # plt.show()

    def close(self):
        return None
