# -*- coding: UTF-8 -*-
import numpy as np
import random
import logging

logging.basicConfig(level=logging.WARNING)


class Action(object):
    def __init__(self):
        self.move = None
        self.collect = None
        self.offloading = []
        self.bandwidth = 0
        self.execution = []


class AgentState(object):
    def __init__(self):
        self.position = None
        self.obs = None


class EdgeDevice(object):
    edge_count = 0

    def __init__(self, obs_r, pos, spd, collect_r, max_buffer_size, movable=True, mv_bt=0, trans_bt=0):  # pos(x,y,h)
        self.no = EdgeDevice.edge_count
        EdgeDevice.edge_count += 1
        self.obs_r = obs_r  # observe radius
        self.init_pos = pos
        self.position = pos
        self.move_r = spd
        self.collect_r = collect_r
        self.mv_battery_cost = mv_bt
        self.trans_battery_cost = trans_bt
        self.data_buffer = {}
        self.max_buffer_size = max_buffer_size
        self.idle = True  # collecting idle
        self.movable = movable
        self.state = AgentState()
        self.action = Action()
        self.done_data = []
        self.offloading_idle = True
        self.total_data = {}
        self.computing_rate = 2e4
        self.computing_idle = True
        self.index_dim = 2
        self.collecting_sensors = []
        self.ptr = 0.2
        self.h = 5
        self.noise = 1e-13
        self.trans_rate = 0

    def move(self, new_move, h):
        if self.idle:
            self.position += new_move
            self.mv_battery_cost += np.linalg.norm(new_move)

    def get_total_data(self):
        total_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        if self.total_data:
            for j, k in enumerate(list(self.total_data.keys())):
                # print(self.total_data[k])
                total_data_state[0, j] = self.total_data[k][0]
                total_data_state[1, j] = self.total_data[k][1]
        return total_data_state

    def get_done_data(self):
        done_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        if self.done_data:
            for m, k in enumerate(self.done_data):
                done_data_state[0, m] = k[0]
                done_data_state[1, m] = k[1]
        return done_data_state

    def data_update(self, pak):
        if pak[1] in self.data_buffer.keys():
            self.data_buffer[pak[1]].append(pak)
        else:
            self.data_buffer[pak[1]] = [pak]

    def edge_exe(self, tmp_size, t=1):  # one-sum local execution
        if not self.total_data:
            return [0] * self.max_buffer_size
        # age update
        for k in self.total_data.keys():
            self.total_data[k][1] += t
        if len(self.done_data) >= self.max_buffer_size:
            return tmp_size
        # process
        if self.total_data and sum(self.action.execution):
            data2process = [[k, d] for k, d in self.total_data.items()]
            self.computing_idle = False
            if np.argmax(self.action.execution) >= len(data2process):
                self.action.execution = [0] * self.max_buffer_size
                self.action.execution[np.random.randint(len(data2process))] = 1
            for i, data in enumerate(data2process):
                if len(self.done_data) >= self.max_buffer_size:
                    break
                # print([i, tmp_size])
                tmp_size[i] += min(self.total_data[data2process[i][0]][0], self.computing_rate * self.action.execution[i] * t)
                self.total_data[data2process[i][0]][0] -= self.computing_rate * self.action.execution[i] * t
                if self.total_data[data2process[i][0]][0] <= 0:
                    self.total_data[data2process[i][0]][0] = tmp_size[i]
                    self.done_data.append(self.total_data[data2process[i][0]])
                    self.total_data.pop(data2process[i][0])
                    tmp_size[i] = 0
        return tmp_size

    def process(self, tmp_size, t=1):  # one-hot local execution
        if not self.total_data:
            return 0
        # age update
        for k in self.total_data.keys():
            self.total_data[k][1] += t
        if len(self.done_data) >= self.max_buffer_size:
            return 0
        # process
        if self.total_data and sum(self.action.execution):
            data2process = [[k, d] for k, d in self.total_data.items()]

            if self.action.execution.index(1) >= len(data2process):
                self.action.execution[self.action.execution.index(1)] = 0
                self.action.execution[np.random.randint(len(data2process))] = 1
                # print(self.action.execution)
            self.computing_idle = False
            tmp_size += min(self.total_data[data2process[self.action.execution.index(
                1)][0]][0], self.computing_rate * t)
            self.total_data[data2process[self.action.execution.index(
                1)][0]][0] -= self.computing_rate * t
            if self.total_data[data2process[self.action.execution.index(1)][0]][0] <= 0:
                self.total_data[data2process[self.action.execution.index(
                    1)][0]][0] = tmp_size
                self.done_data.append(self.total_data[data2process[self.action.execution.index(
                    1)][0]])
                self.total_data.pop(data2process[self.action.execution.index(
                    1)][0])
                tmp_size = 0
        return tmp_size


def agent_com(agent_list):
    age_dict = {}
    for u in agent_list:
        for k, v in u.data_buffer.items():
            if k not in age_dict:
                age_dict[k] = v[-1][1]
            elif age_dict[k] > v[-1][1]:
                age_dict[k] = v[-1][1]
    return age_dict


class Sensor(object):
    sensor_cnt = 0

    def __init__(self, pos, data_rate, bandwidth, max_ds, lam=0.5, weight=1):
        self.no = Sensor.sensor_cnt
        Sensor.sensor_cnt += 1
        self.position = pos
        self.weight = weight
        self.data_rate = data_rate  # generate rate
        self.bandwidth = bandwidth
        self.trans_rate = 8e3  # to be completed
        self.data_buffer = []
        self.max_data_size = max_ds
        self.data_state = bool(len(self.data_buffer))
        self.collect_state = False
        self.lam = lam
        self.noise_power = 1e-13 * self.bandwidth
        self.gen_threshold = 0.3

    def data_gen(self, t=1):
        # update age
        if self.data_buffer:
            for i in range(len(self.data_buffer)):
                self.data_buffer[i][1] += t
        new_data = self.data_rate * np.random.poisson(self.lam)
        # new_data = min(new_data, self.max_data_size)
        if new_data >= self.max_data_size or random.random() >= self.gen_threshold:
            return
        if new_data:
            self.data_buffer.append([new_data, 0, self.no])
            self.data_state = True


collecting_channel_param = {'suburban': (4.88, 0.43, 0.1, 21),
                            'urban': (9.61, 0.16, 1, 20),
                            'dense-urban': (12.08, 0.11, 1.6, 23),
                            'high-rise-urban': (27.23, 0.08, 2.3, 34)}

collecting_params = collecting_channel_param['urban']
a = collecting_params[0]
b = collecting_params[1]
yita0 = collecting_params[2]
yita1 = collecting_params[3]
carrier_f = 2.5e9


def collecting_rate(sensor, agent):
    d = np.linalg.norm(np.array(sensor.position) - np.array(agent.position))
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / d) - a)))
    L = Pl * yita0 + yita1 * (1 - Pl)
    gamma = agent.ptr_col / (L * sensor.noise_power**2)
    rate = sensor.bandwidth * np.log2(1 + gamma)
    return rate


def data_collecting(sensors, agent, hovering_time):
    for k in agent.total_data.keys():
        agent.total_data[k][1] += 1
    if agent.idle and (len(agent.total_data.keys()) < agent.max_buffer_size):
        # obs_sensor = []
        data_properties = []
        # for k in agent.data_buffer.keys():
        #     for i, d in enumerate(agent.data_buffer[k]):
        #         agent.data_buffer[k][i][1] += 1
        for sensor in sensors:
            if not sensor.data_buffer:
                continue
            if (np.linalg.norm(np.array(sensor.position) - np.array(agent.position)) <= agent.collect_r) and not(sensor.collect_state) and not(sensor.no in agent.total_data.keys()):
                sensor.collect_state = True
                agent.collecting_sensors.append(sensor.no)
                agent.idle = False
                if len(agent.total_data.keys()) >= agent.max_buffer_size:
                    continue
                # obs_sensor.append(sensor)
                # if not (sensor.no in agent.data_buffer.keys()):
                #     agent.data_buffer[sensor.no] = []
                tmp_size = 0
                # trans_rate = collecting_rate(sensor, agent)
                for data in sensor.data_buffer:
                    tmp_size += data[0]
                    # data[1] += tmp_size / self.trans_rate  # age update
                if sensor.no in agent.data_buffer.keys():
                    agent.data_buffer[sensor.no].append(tmp_size)
                else:
                    agent.data_buffer[sensor.no] = [tmp_size]
                data_properties.append(tmp_size / sensor.trans_rate)
                agent.total_data[sensor.no] = [tmp_size, sensor.data_buffer[-1][1], sensor.no]
                # agent.total_data[sensor.no] = [tmp_size, np.average([x[1] for x in sensor.data_buffer]), sensor.no]
                sensor.data_buffer = []

        if data_properties:
            hovering_time = max(data_properties)
            # print([data_properties, hovering_time])
            return hovering_time
        else:
            return 0
    # finish collection
    elif not agent.idle:
        hovering_time -= 1
        if hovering_time <= 0:
            agent.idle = True
            for no in agent.collecting_sensors:
                sensors[no].collect_state = False
            agent.collecting_sensors = []
            hovering_time = 0
        return hovering_time
    else:
        return 0


def offloading(agent, center_pos, t=1):
    if not agent.done_data:
        #print('no done')
        return (False, {})
    for data in agent.done_data:
        data[1] += t

    if sum(agent.action.offloading):
        if agent.action.offloading.index(1) >= len(agent.done_data):
            agent.action.offloading[agent.action.offloading.index(1)] = 0
            agent.action.offloading[np.random.randint(len(agent.done_data))] = 1
        agent.offloading_idle = False
        dist = np.linalg.norm(np.array(agent.position) - np.array(center_pos))
        agent.trans_rate = trans_rate(dist, agent)  # to be completed
    else:
        return False, {}
    # print(agent.done_data)
    # print(agent.action.offloading)
    agent.done_data[agent.action.offloading.index(1)][0] -= agent.trans_rate * t
    if agent.done_data[agent.action.offloading.index(1)][0] <= 0:
        sensor_indx = agent.done_data[agent.action.offloading.index(1)][2]
        sensor_aoi = agent.done_data[agent.action.offloading.index(1)][1]
        sensor_data = agent.data_buffer[sensor_indx][0]
        del agent.data_buffer[sensor_indx][0]
        del agent.done_data[agent.action.offloading.index(1)]
        # return finish flag & total data
        agent.offloading_idle = True
        return True, {sensor_indx: [sensor_data, sensor_aoi]}
    return False, {}


def trans_rate(dist, agent):  # to be completed
    W = 1e6 * agent.action.bandwidth
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / dist) - a)))
    fspl = (4 * np.pi * carrier_f * dist / (3e8))**2
    L = Pl * fspl * 10**(yita0 / 20) + 10**(yita1 / 20) * fspl * (1 - Pl)
    rate = W * np.log2(1 + agent.ptr / (L * agent.noise * W))
    print('agent-{} rate: {},{},{},{},{}'.format(agent.no, dist, agent.action.bandwidth, Pl, L, rate))
    return rate


class MEC_world(object):
    def __init__(self, map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size=1, sensor_lam=0.5):
        self.agents = []
        self.sensors = []
        self.map_size = map_size
        self.center = (map_size / 2, map_size / 2)
        self.sensor_count = sensor_num
        self.agent_count = agent_num
        self.max_buffer_size = max_size
        sensor_bandwidth = 1000
        max_ds = sensor_lam * 2
        data_gen_rate = 1
        self.offloading_slice = 1
        self.execution_slice = 1
        self.time = 0
        self.DS_map = np.zeros([map_size, map_size])
        self.DS_state = np.ones([map_size, map_size, 2])
        self.hovering_list = [0] * self.agent_count
        self.tmp_size_list = [0] * self.agent_count
        # [self.tmp_size_list.append([0] * self.max_buffer_size) for i in range(self.agent_count)]
        self.offloading_list = []
        self.finished_data = []
        self.obs_r = obs_r
        self.move_r = speed
        self.collect_r = collect_r
        self.sensor_age = {}
        # random.seed(7)
        self.sensor_pos = [random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num),
                           random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num)]
        # self.sensor_pos = [random.choices([i for i in range(int(0.1 * self.map_size), int(0.5 * self.map_size))], k=int(sensor_num / 2)) + random.choices(
        #     [i for i in range(int(0.5 * self.map_size), int(0.9 * self.map_size))], k=int(sensor_num / 2)), random.choices([i for i in range(int(0.1 * self.map_size), int(0.9 * self.map_size))], k=sensor_num)]
        for i in range(sensor_num):
            self.sensors.append(
                Sensor(np.array([self.sensor_pos[0][i], self.sensor_pos[1][i]]), data_gen_rate, sensor_bandwidth, max_ds, lam=sensor_lam))
            self.sensor_age[i] = 0
            self.DS_map[self.sensor_pos[0][i], self.sensor_pos[1][i]] = 1
        self.agent_pos_init = [random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num),
                               random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num)]

        for i in range(agent_num):
            self.agents.append(
                EdgeDevice(self.obs_r, np.array([self.agent_pos_init[0][i], self.agent_pos_init[1][i]]), speed, collect_r, self.max_buffer_size))

    def step(self):
        # update sensor age
        for k in self.sensor_age.keys():
            self.sensor_age[k] += 1
        # data generation & DS_state update
        logging.info("data generation")
        for sensor in self.sensors:
            sensor.data_gen()
            if sensor.data_buffer:
                data_size = sum(i[0] for i in sensor.data_buffer)
                # update data source state, note that the (x,y) is reversed to the matrix index (i,j)
                self.DS_state[sensor.position[1], sensor.position[0]] = [
                    data_size, sensor.data_buffer[0][1]]

        # edge process  offloading collect
        logging.info("edge operation")
        age_dict = {}
        for i, agent in enumerate(self.agents):
            # edge process
            self.tmp_size_list[i] = agent.process(self.tmp_size_list[i])
            # offloading
            finish_flag, data_dict = offloading(agent, self.center)
            # update reward state
            # print([i, finish_flag, data_dict])
            if finish_flag:
                for sensor_id, data in data_dict.items():
                    self.finished_data.append([data[0], data[1], sensor_id])
                    if sensor_id in age_dict.keys():
                        age_dict[sensor_id].append(data[1])
                    else:
                        age_dict[sensor_id] = [data[1]]
                    # self.sensor_age[sensor_id] -=data[1]
            # collecting
            self.hovering_list[i] = data_collecting(self.sensors, agent, self.hovering_list[i])
            # print(self.hovering_list[i])
        for id in age_dict.keys():
            self.sensor_age[id] = sorted(age_dict[id])[0]
        print('hovering:{}'.format(self.hovering_list))
