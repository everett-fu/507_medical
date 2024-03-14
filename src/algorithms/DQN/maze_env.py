import random
import numpy as np
import time
import sys

import pandas as pd
import data_set

class EV:
    # 初始化电车信息，输入：电车数量，节点数量，充电路段数量
    def __init__(self, number_evs, number_nodes, number_charge):
        self.number_evs = number_evs
        self.number_nodes = number_nodes
        self.number_charge = number_charge
        # 初始化地图
        rows = self.number_nodes
        self.action_space = []  # 可选路段集合
        # rows = self.number_nodes + self.number_evs + 1
        cols = rows
        for i in range(rows):
            for j in range(cols):
                self.action_space.append((i, j))
        self.n_actions = len(self.action_space)
        self.n_features = 3
        print(self.n_actions, self.n_features, self.action_space)

    # 初始化交通地图
    # 返回：距离矩阵，是否为充电矩阵，速度矩阵，可选路段集合
    def init_graph(self):
        file_path = '../../script/data/changshu.xls'
        data = pd.read_excel(file_path)
        random_speed = -1
        random_point = -1
        n = self.number_nodes
        m = self.number_charge
        adjacency_matrix_distance, adjacency_matrix_charge, adjacency_matrix_speed = \
            data_set.build_adjacency_matrix(data, random_speed, random_point, n, m)
        return adjacency_matrix_distance, adjacency_matrix_charge, adjacency_matrix_speed, self.action_space

    # 初始化电动车信息
    # 返回：初始节点，目标节点，初始电量，初始dealline
    def init_evs_info(self):
        evs_info = data_set.ini_evs_info(self.number_evs, self.number_nodes)
        return evs_info

    # 重置环境到初始状态
    def reset(self, start, end, power, deadline):
        segment_0, segment_1, E_0, t_0= start, end, power, deadline
        return [segment_0, segment_1, E_0, t_0]

    # 执行动作
    def step(self, starting, observation, action_space, action, graph_speed, graph, ending, path_id):
        # 当前状态：当前路段，目标地址，剩余电量，剩余deadline
        segment1, segment2, E1, t1 = observation[0], observation[1], observation[2], observation[3]
        s_ = [0, 0, 0, 0]
        done = False
        start = action_space[segment1][0]   # 当前路段的起点
        end = action_space[segment1][1]     # 当前路段的终点
        r_e_a = graph_speed[start][end]     # EV行驶速度km/h
        r_b = 40    # speed of bus 40km/h
        charging_power = 100    # 公共巴士充电功率100kw
        E_e_max = 100   # 电车的电池容量kWh
        gamma = 1   # 电车的能耗，每公里消耗1kWh
        reward = -gamma * graph[start][end]     # 走完该路段的能耗
        E_ = E1 - gamma * graph[start][end]     # 剩余电量
        if r_e_a == 0:
            t_ = -1000
        else:
            t_ = t1 - graph[start][end] / r_e_a
        visited_path = []
        for idd in path_id:
            visited_path.append(action_space[idd])
        visited_path = np.array(visited_path).flatten()  # 二维列表转换为一维列表
        vp = set(visited_path)  # 路径访问过的点集合
        if end == ending:  # 到达终点
            s_ = [action, E_, t_]
            print("[ev_env.step]终止调度，到达终点了，s_=", s_)
            done = True  # 终止调度
        else:
            if action_space[action][0] == end and (action_space[action][1] in vp) == False and (
                    action_space[action][1] == starting) == False:
                if E_ <= 0 or t_ <= 0:
                    # 剩余电量为0或者deadline为0，退出，失败。
                    print("[ev_env.step]不满足约束了,E_,t_=", E_, t_)
                    reward = 0
                    done = True  # 终止调度
                else:  # 继续调度
                    s_ = [action, E_, t_]
                    print("[ev_env.step]继续调度s_=", s_)
            else:  # 当前路段无法使用
                done = True
                reward = 0

        return s_, reward, done