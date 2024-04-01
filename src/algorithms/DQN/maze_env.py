import numpy as np
import pandas as pd

import src.algorithms.data_set as data_set

# import os

class EV:
    # 初始化电车信息，输入：电车数量，节点数量，充电路段数量
    def __init__(self, number_evs, number_nodes, number_charge):
        # print("当前工作目录:", os.getcwd())

        self.number_evs = number_evs
        self.number_nodes = number_nodes
        self.number_charge = number_charge
        # 初始化地图
        # 这个代码是从已有的地图中选一个点作为起始点，所以动作数量是地图点位
        rows = self.number_nodes
        self.action_space = []  # 可选路段集合
        # 这个代码是从已有的地图以外选一个点作为起始点，所以动作数量是地图点位加上电车起始点的数量
        # rows = self.number_nodes + self.number_evs + 1
        cols = rows
        for i in range(rows):
            for j in range(cols):
                self.action_space.append((i, j))
        self.n_actions = len(self.action_space)
        self.n_features = 4
        print(self.n_actions, self.n_features, self.action_space)

    # 初始化交通地图
    # 返回：距离矩阵，是否为充电矩阵，速度矩阵，可选路段集合
    # def init_graph(self):
    #     adjacency_matrix_distance = np.array([[0, 30.21940343, 25.76162458, 50.48488941, 9.1017145],
    #                                          [30.21940343, 0, 10.99231348, 36.99807237, 25.1244118],
    #                                          [25.76162458, 10.99231348, 0, 29.55209479, 18.20263773],
    #                                          [50.48488941, 36.99807237, 29.55209479, 0, 41.38846447],
    #                                          [9.1017145, 25.1244118, 18.20263773, 41.38846447, 0]])
    #     # 充电情况数据
    #     adjacency_matrix_charge = np.array([[0, 0, 0, 0, 0],
    #                        [0, 0, 0, 1, 1],
    #                        [0, 0, 0, 0, 0],
    #                        [0, 1, 0, 0, 1],
    #                        [0, 1, 0, 1, 0]])
    #     # 速度数据
    #     adjacency_matrix_speed = np.array([[59, 59, 45, 53, 48],
    #                       [56, 40, 54, 50, 51],
    #                       [59, 42, 58, 51, 59],
    #                       [60, 44, 60, 44, 46],
    #                       [45, 55, 43, 41, 46]])
    #
    #     return adjacency_matrix_distance, adjacency_matrix_charge, adjacency_matrix_speed, self.action_space
    def init_graph(self):
        file_path = '../script/data/changshu.xls'
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
    # def init_evs_info(self):
    #     evs_info = []
    #     for i in range(self.number_evs):
    #         start = 2
    #         end = 0
    #         power = 54.42
    #         deadline = 4.97
    #         evs_info.append([start, end, power, deadline])
    #     return evs_info
    def init_evs_info(self):
        evs_info = data_set.ini_evs_info(self.number_evs, self.number_nodes)
        return evs_info

    # 重置环境到初始状态
    def reset(self, start, end, power, deadline):
        segment_0, segment_1, E_0, t_0 = start, end, power, deadline
        return [segment_0, segment_1, E_0, t_0]

    # 执行动作
    # 输入电车起始点，当前状态，动作空间，动作，距离矩阵，充电矩阵，速度矩阵，走过路径的编号
    # 返回：下一状态，奖励，是否结束
    def step(self, starting, observation, action_space, action, graph, graph_charge, graph_speed, path_id):
        # 当前状态：当前路段，目标地址，剩余电量，剩余deadline
        segment1, segment2, E1, t1 = observation[0], observation[1], observation[2], observation[3]
        s_ = [0, 0, 0, 0]                       # 下一状态
        done = False                            # 是否结束
        start = action_space[segment1][0]       # 当前路段的起点，即从A出发到B的A
        end = action_space[segment1][1]         # 当前路段的终点，即从A出发到B的B
        speed = graph_speed[start][end]         # 当前路段的行驶速度
        gamma = 0.1                               # 能耗，每公里消耗1kWh
        charge_power = 2                      # 充电功率，每小时100kWh

        charge = 0                              # 电量变化值
        # 如果当前路段是充电路段，加上充电量减去耗电量
        if graph_charge[start][end] == 1:
            charge += charge_power * graph[start][end] / speed - graph[start][end] * gamma
        else:
            charge += -graph[start][end] * gamma
        # 更新剩余电量
        E_ = E1 + charge
        # 电量最大为100
        if E_ > 100:
            E_ = 100

        # 更新奖励函数
        reward = charge

        # 更新剩余时间
        if speed == 0:
            t_ = -1000
        else:
            t_ = t1 - graph[start][end] / speed
            # print(f"start:{start}, end:{end}, speed:{speed}")
            # print(f"time1:{graph[start][end] / speed}, time2:{t_}")

        # 提取出访问过的点级
        visited_path = []
        for idd in path_id:
            visited_path.append(action_space[idd])
        visited_path = np.array(visited_path).flatten()
        vp = set(visited_path)

        # 如果到达终点
        if end == segment2:
            s_ = [action, segment2, E_, t_]
            reward = 1
            done = True
        else:
            # 如果当前路段不是从终点出发，且不在访问过的点集合中，且没有绕回到起点
            if action_space[action][0] == end and (action_space[action][1]) not in vp and (action_space[action][1] != starting):
                # 如果电量为0或者时间为0，退出，失败
                if E_ <= 0 or t_ <= 0:
                    reward = -100
                    done = True
                else:
                    s_ = [action, segment2, E_, t_]
            else:
                done = True
                reward = -100

        return s_, reward, done