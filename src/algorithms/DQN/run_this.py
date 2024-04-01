import json

from src.algorithms.DQN.maze_env import EV
from src.algorithms.DQN.RL_brain import DeepQNetwork
import time

'''
这个函数用于运行DQN
该函数有个输入，分别是：
    + source_roadsegment: 起始路段编号
    + end_roadsegment: 结束路段编号
    + E_idx_0: 初始电量
    + deadline_idx_0: 截止时间
    + graph: 距离矩阵
    + graph_charge: 是否为充电矩阵
    + graph_speed: 速度矩阵
    + action_space: 可选路段集合
    
该函数的输出是三个值，分别是：
    + final_path: 最终路径
    + max_info[1]: 剩余电量
    + max_info[2]: 剩余截止时间
'''
def run_ev(source_roadsegment, end_roadsegment, E_idx_0, deadline_idx_0, graph, graph_charge, graph_speed, action_space):
    # TODO 重写这个函数
    step = 0
    # ev_id = 1
    # evs_info = env.init_evs_info()
    # graph, graph_speed, action_space = env.init_graph()
    # 返回：交通图，路段列表，EV初始状态（3个参数）
    # print("[run_this] action space", action_space)
    print(f'[runt this]当前DQN的起始路段编号：{source_roadsegment}, 动作空间： {len(action_space)}')
    print(
        f'[runt this]当前DQN的起始路段编号：{source_roadsegment}, 起始路段元祖(起点，终点)：{action_space[source_roadsegment]}')
    starting = action_space[source_roadsegment * len(graph) + source_roadsegment][1]
    ending = end_roadsegment
    max_energy = 0
    max_info = []
    final_path = []
    for episode in range(400):
        graph, graph_charge, graph_speed, action_space = graph, graph_charge, graph_speed, action_space
        print('----------第%d幕计算' % episode)
        observation = env.reset(source_roadsegment, end_roadsegment, E_idx_0, deadline_idx_0)
        # 初始状态:起点路段编号，初始剩余电量kWh，初始deadline,hour
        print('[run_this]当前EV的起始状态：', observation)
        path = []
        path_id = []
        while True:
            action = RL.choose_action(observation, action_space, path_id)  # 选择一条路段(i,j)
            # print('被选中的路段：',action)
            observation_, reward, done = (
                env.step(starting, observation, action_space, action, graph, graph_charge, graph_speed, path_id))
            print("[run_this]", observation_, reward, done)
            RL.store_transition(observation, action, reward, observation_)
            # 经验回放：experience replay
            if (step > 200) and (step % 5 == 0):
                # print("[run_this]启动神经网络学习")
                RL.learn()
            if done:
                print("\n [run_this] Game over.")
                break
            observation = observation_
            step += 1
            path.append(action_space[action])  # 记录走过的路段（起点，终点）
            path_id.append(action)  # 记录走过的路段编号
            print("\n [run_this] 当前时间步step=", step)

        if path != []:
            path.insert(0, (starting, path[0][0]))
            # 插入以起点为终点的路段（自己到自己）
            if path[len(path) - 1][1] != ending:
                path.append((path[len(path) - 1][1], ending))
                # 插入到达终点的路段
            print("\n [run_this] path=", path)
            print(f'\n [run_this] 剩余电量：{observation[2]}，剩余截止时间：{observation[3]}')
            if max_energy <= observation[2]:
                max_info = observation
                final_path = path
        else:
            print("\n [run_this] path is invalid.")
    print('game over!')
    if len(final_path) > 0:
        print("\n [run_this] 当前车辆的起点，终点的序号", starting, ending)
        print("\n [run_this] final_path=", final_path)
        print(f'\n [run_this] 最终路径的剩余电量:{max_info[2]}，剩余截止时间:{max_info[3]}')
        return final_path, max_info[2], max_info[3]
    else:
        print("\n [run_this] final path is invalid.")
        return final_path, 0, 0


def save_to_meus_json(data, filename):
    with open('./data/{}'.format(filename), mode='a', encoding='utf-8') as f_meus:
        json.dump(data, f_meus)


def compute_social_efficiency(total_energy, total_length):
    lbd_s = 0.16  # 充电站售电价格，单位美元
    lbd_b = 0.13  # 新能源发电站售电价格，单位美元
    total_newpower = 0  # 发电站总电量
    gamma = 0.1  # 单位公里电能耗，单位kwh
    # for id in range(len(newpower_value)):
    #     total_newpower = total_newpower + newpower_value[id]
    v_F = lbd_s * total_energy
    return v_F - gamma * total_length * lbd_s


def compute_length(graph, path):
    length_path = 0
    print(path)
    for item in path:
        print(item, item[0], item[1])
        length_path = length_path + graph[int(item[0])][int(item[1])]
    return length_path


if __name__ == "__main__":
    # number_evs = 3  # 电车数量
    # 设置最大循环次数
    max_counter = 20

    # 电车剩余电量
    power = []

    # 节点数
    for number_nodes in range(5, 11):
        number_charge = int(number_nodes * 2 / 3)   # 充电节点数量，数目为节点数的2/3

        # 本次节点电车剩余电量
        total_power = []
        # 电车数，从5开始，每次增加5
        for number_evs in range(5,16, 5):

            # 初始化节点数量
            # number_nodes = 5  # 节点数量
            # number_charge = 3  # 充电节点数量
            gamma = 1  # 单位公里电能耗，单位kwh
            with open('./data/result_5.txt', mode='a', encoding='utf-8') as f:
                f.write("地图编号 剩余电量 路径 路径长度 电车序号")
                f.write("\n")
            # counter = 1
            # 跑完的总次数
            total_winners = 0
            total_social_effiency = 0
            # 程序开始时间
            start_time = time.perf_counter()
            # while counter < max_counter:

            # 每一轮电车的剩余电量
            counter_power = 0

            # 跑max_counter轮算平均值
            for counter in range(1, max_counter):
                total_payment_cost_ratio = 0
                total_newpower_value = 0
                total_length = 0
                total_cost = 0

                # 生成电车信息
                env = EV(number_evs, number_nodes, number_charge)
                # 初始化电车信息
                evs_info = env.init_evs_info()
                # 初始化交通地图
                graph, graph_charge, graph_speed, action_space = env.init_graph()
                # 初始化DQN
                RL = DeepQNetwork(env.n_actions, env.n_features,
                                  learning_rate=0.01,
                                  reward_decay=0.9,
                                  e_greedy=0.9,
                                  replace_target_iter=200,
                                  memory_size=2000,
                                  output_graph=False
                                  )
                print("环境参数初始化结束")

                # 记录最大电量电车信息
                max_path = []
                max_E = 0
                max_evid = -1
                max_ev_ids = []  # 记录被选中的所有电车序号
                max_ev_length = []  # 记录被选中的所有电车路径长度
                # 遍历每辆电车
                # while ev_idx < number_evs:
                for ev_idx in range(number_evs):
                    # 如果电车已经被选中，跳过
                    if ev_idx in max_ev_ids:
                        break
                    else:
                        ev_idx_info = evs_info[ev_idx]  # 当前电车的信息
                        source_roadsegment = ev_idx_info[0]  # 起点所在路段编号
                        end_roadsegment = ev_idx_info[1]  # 终点所在路段编号
                        E_idx_0 = ev_idx_info[2]  # 当前电车的初始电量
                        deadline_idx_0 = ev_idx_info[3]  # 当前电车的截止时间

                        # 运行，返回路径，剩余电量，剩余截止时间
                        path_s_nps, E_s_nps, deadline_s_nps = run_ev(source_roadsegment, end_roadsegment, E_idx_0,
                                                                     deadline_idx_0, graph, graph_charge, graph_speed,
                                                                     action_space)
                        print("当前电车信息：", source_roadsegment, end_roadsegment, E_idx_0, deadline_idx_0)
                        print("电车序号：", ev_idx)
                        print("电车起点到终点的路径，剩余电量，剩余截止时间", path_s_nps, E_s_nps, deadline_s_nps)

                    # 如果有路径
                    if len (path_s_nps) > 0:
                        completed_path = path_s_nps
                        # 添加剩余电量
                        counter_power += E_s_nps

                        # 如果剩余电量大于最大电量，记录最大路径，最大电量，最大电车序号
                        if E_s_nps > max_E:
                            max_path = completed_path
                            max_E = E_s_nps
                            max_evid = ev_idx

                # 记录最大电量电车信息
                if max_E > 0 and max_evid > -1:
                    # 计算最大电量电车的路径长度
                    length_path = compute_length(graph, max_path)
                    # 计算总路径长度
                    total_length = total_length + length_path
                    # total_cost = total_cost + gamma * length_path / newpower_value[nps_idx]
                    max_ev_ids.append(max_evid)
                    max_ev_length.append(length_path)
                    with open('./data/result_5.txt', mode='a', encoding='utf-8') as f:
                        f.write(str(counter)+' ')#地图编号
                        f.write(str(max_E)+' ')#适用于当前发电站的最佳路径的剩余电量
                        for item in max_path:
                            f.write(str(item))
                        f.write(' '+str(length_path))
                        f.write(' '+str(max_ev_ids))
                        f.write(' '+str(max_ev_length))
                        f.write("\n")
                    total_energy = total_newpower_value + max_E
                    # total_newpower_value = total_newpower_value + max_E
                    total_winners = total_winners + 1
            total_social_effiency = total_energy

            # 跑完所有轮的总电量的平均值
            counter_power /= max_counter

                # total_social_effiency = total_social_effiency+compute_social_efficiency(total_newpower_value, total_length)

            # #统计counter次实验的均值
            # number_winners = total_winners/(counter-1)
            # social_efficiency = total_social_effiency/(counter-1)
            # payment_cost_ratio = total_payment_cost_ratio / (counter-1)
            # # 程序结束时间
            # end_time = time.perf_counter()
            # runing_time = (end_time-start_time)/(counter-1)
            # data = {
            #     "number_of_evs":number_evs,
            #     "number_of_winners":number_winners,
            #     "social_efficiency":social_efficiency,
            #     "payment_cost_ratio":payment_cost_ratio,
            #     "running time": runing_time
            # }
            # save_to_meus_json(data,'meus_5.json')

            # 跑完该节点数的总电量
            total_power.append(counter_power)
        # 跑完所有电车数的总电量
        power.append(total_power)

    print(power)
