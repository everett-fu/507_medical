import json

from src.algorithms.DQN.maze_env import EV
from src.algorithms.DQN.RL_brain import DeepQNetwork
import time

def run_ev(source_roadsegment,nps_idx_location,E_idx_0,deadline_idx_0,graph, graph_speed, action_space):
    step = 0
    # ev_id = 1
    # evs_info = env.init_evs_info()
    # graph, graph_speed, action_space = env.init_graph()
    # 返回：交通图，路段列表，EV初始状态（3个参数）
    # print("[run_this] action space", action_space)
    print("[runt this]当前DQN的起始路段编号，动作空间", source_roadsegment,len(action_space))
    print("[runt this]当前DQN的起始路段编号，起始路段元祖（起点，终点）",source_roadsegment,action_space[source_roadsegment])
    starting = action_space[source_roadsegment][1]
    ending = nps_idx_location
    max_energy = 0
    max_info = []
    final_path = []
    for episode in range(400):
        graph, graph_speed, action_space = graph,graph_speed,action_space
        print('----------第%d幕计算' %episode)
        observation = env.reset(source_roadsegment,E_idx_0,deadline_idx_0)
        #初始状态:起点路段编号，初始剩余电量kWh，初始deadline,hour
        print('[run_this]当前EV的起始状态：', observation)
        path = []
        path_id=[]
        while True:
            action = RL.choose_action(observation,action_space,path_id)#选择一条路段(i,j)
            # print('被选中的路段：',action)
            observation_, reward, done = (
                env.step(starting,observation,action_space,action,graph_speed,graph,ending,path_id))
            print("[run_this]",observation_,reward,done)
            RL.store_transition(observation, action, reward, observation_)
            #经验回放：experience replay
            if (step > 200) and (step % 5 == 0):
               # print("[run_this]启动神经网络学习")
               RL.learn()
            if done:
               print("\n [run_this] Game over.")
               break
            observation = observation_
            step += 1
            path.append(action_space[action])#记录走过的路段（起点，终点）
            path_id.append(action)#记录走过的路段编号
            print("\n [run_this] 当前时间步step=", step)

        if path != []:
            path.insert(0,(starting,path[0][0]))
            #插入以起点为终点的路段（自己到自己）
            if path[len(path)-1][1]!=ending:
                path.append((path[len(path)-1][1],ending))
                # 插入到达终点的路段
            print("\n [run_this] path=", path)
            print("\n [run_this] 剩余电量，剩余截止时间", observation[1], observation[2])
            if max_energy<=observation[1]:
                max_info = observation
                final_path = path
        else:
            print("\n [run_this] path is invalid.")
    print('game over!')
    if len(final_path)>0:
        print("\n [run_this] 当前车辆的起点，终点的序号", starting, ending)
        print("\n [run_this] final_path=", final_path)
        print("\n [run_this] 最终路径的剩余电量，剩余截止时间=", max_info[1], max_info[2])
        return final_path, max_info[1], max_info[2]
    else:
        print("\n [run_this] final path is invalid.")
        return final_path,0,0

def save_to_meus_json(data,filename):
    with open('./data/{}'.format(filename), mode='a', encoding='utf-8') as f_meus:
        json.dump(data, f_meus)

def compute_social_efficiency(total_energy, total_length):
    lbd_s = 0.16#充电站售电价格，单位美元
    lbd_b = 0.13#新能源发电站售电价格，单位美元
    total_newpower= 0#发电站总电量
    gamma = 0.1#单位公里电能耗，单位kwh
    # for id in range(len(newpower_value)):
    #     total_newpower = total_newpower + newpower_value[id]
    v_F=lbd_s * total_energy
    return v_F-gamma*total_length*lbd_s

def compute_length(graph,path):
    length_path = 0
    print(path)
    for item in path:
        print(item,item[0],item[1])
        length_path = length_path + graph[int(item[0])][int(item[1])]
    return length_path

if __name__=="__main__":
    number_evs =5
    max_counter = 100
    while number_evs <= 15:
        number_newpower=3
        number_nodes=3
        gamma = 0.1  # 单位公里电能耗，单位kwh
        with open('./data/result_5.txt', mode='a', encoding='utf-8') as f:
            f.write("地图编号 发电站 剩余电量 路径 路径长度 电车序号")
            f.write("\n")
        counter = 1
        total_winners = 0
        total_social_effiency = 0
        start_time = time.perf_counter()
        while counter < max_counter:
            total_payment_cost_ratio = 0
            total_newpower_value = 0
            total_length = 0
            total_cost = 0
            env = EV(number_evs,number_newpower,number_nodes)
            evs_info = env.init_evs_info()
            graph, graph_speed, action_space = env.init_graph()
            newpower_value,nps_idxs = env.init_newpower()
            RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate = 0.01,
                          reward_decay = 0.9,
                          e_greedy = 0.9,
                          replace_target_iter = 200,
                          memory_size = 2000,
                          output_graph=False
                          )
            print("环境参数初始化结束")
            nps_idx = 0#发电站编号初始值

            while nps_idx < number_newpower:
            #遍历每个新能源发电站
                nps_idx_curr = nps_idxs[nps_idx]#当前发电站的坐标
                max_path = []
                max_E = 0
                ev_idx = 0
                max_evid = -1
                max_ev_ids = []  # 记录被选中的所有电车序号
                max_ev_length = []#记录被选中的所有电车路径长度
                while ev_idx < number_evs:
                #遍历每辆电车
                    if ev_idx in max_ev_ids:
                        break
                    else:
                        ev_idx_info = evs_info[ev_idx]#当前电车的信息
                        source_roadsegment=ev_idx_info[0]#起点所在路段编号
                        E_idx_0=ev_idx_info[1]#当前电车的初始电量
                        deadline_idx_0=ev_idx_info[2]#当前电车的截止时间
                        path_s_nps,E_s_nps,deadline_s_nps=run_ev(source_roadsegment,nps_idx_curr,E_idx_0,deadline_idx_0,graph, graph_speed, action_space)
                        print("当前电车信息：", source_roadsegment, E_idx_0, deadline_idx_0)
                        print("发电站序号：", nps_idx)
                        print("电车序号：",ev_idx)
                        print("电车起点到发电站的路径，剩余电量，剩余截止时间",path_s_nps,E_s_nps,deadline_s_nps)
                        if len(path_s_nps)>0:
                            nps_idex_roadsegment = action_space.index(path_s_nps[len(path_s_nps)-1])#发电站所在路段编号
                            E_nps_L_0=E_s_nps+newpower_value[nps_idx]
                            deadline_nps_L_0=deadline_s_nps
                            print("当前电车信息：", nps_idex_roadsegment, E_nps_L_0, deadline_nps_L_0)
                            path_nps_L, E_nps_L, deadline_nps_L = run_ev(nps_idex_roadsegment, 0, E_nps_L_0, deadline_nps_L_0,
                                                                     graph, graph_speed, action_space)

                            print("发电站到充电站L的路径，剩余电量，剩余截止时间", path_nps_L, E_nps_L, deadline_nps_L)
                        if len(path_s_nps) > 0 and len(path_nps_L)>0:
                            completed_path = path_s_nps + path_nps_L
                            print("电车经过发电站到达充电站的完整路径：",completed_path)
                            if E_nps_L>max_E:
                                max_path=completed_path
                                max_E=E_nps_L
                                max_evid = ev_idx
                    ev_idx = ev_idx + 1
                if max_E > 0 and max_evid > -1:
                    length_path = compute_length(graph, max_path)
                    total_length = total_length + length_path
                    total_cost = total_cost + gamma * length_path / newpower_value[nps_idx]
                    max_ev_ids.append(max_evid)
                    max_ev_length.append(length_path)
                    with open('./data/result_5.txt', mode='a', encoding='utf-8') as f:
                        f.write(str(counter)+' ')#地图编号
                        f.write(str(nps_idx)+' ')#发电站编号
                        f.write(str(max_E)+' ')#适用于当前发电站的最佳路径的剩余电量
                        for item in max_path:
                            f.write(str(item))
                        f.write(' '+str(length_path))
                        f.write(' '+str(max_ev_ids))
                        f.write(' '+str(max_ev_length))
                        f.write("\n")
                    total_newpower_value = total_newpower_value + newpower_value[nps_idx]
                    total_winners = total_winners + 1
                nps_idx = nps_idx + 1

            total_social_effiency = total_social_effiency+compute_social_efficiency(total_newpower_value, total_length)

#支付决策阶段
            ev_pe = 0

            total_p_e = 0
            lbd_s = 0.16  # 充电站售电价格，单位美元
            while ev_pe < len(max_ev_ids):#计算每辆电车的支付p_e,在没有当前ev的集合中计算winner集合
                nps_idx = 0  # 发电站编号初始值
                curr_ev = max_ev_ids[ev_pe]#当前电车，计算支付值
                total_cost_not_e = 0
                while nps_idx < number_newpower:
                    # 遍历每个新能源发电站
                    nps_idx_curr = nps_idxs[nps_idx]  # 当前发电站的坐标
                    max_path = []
                    max_E = 0
                    ev_idx = 0
                    while ev_idx < number_evs  and ev_idx != curr_ev:# 遍历除了当前电车的每辆电车
                        ev_idx_info = evs_info[ev_idx]  # 当前电车的信息
                        source_roadsegment = ev_idx_info[0]  # 起点所在路段编号
                        E_idx_0 = ev_idx_info[1]  # 当前电车的初始电量
                        deadline_idx_0 = ev_idx_info[2]  # 当前电车的截止时间
                        path_s_nps, E_s_nps, deadline_s_nps = run_ev(source_roadsegment, nps_idx_curr, E_idx_0, deadline_idx_0,
                                                                     graph, graph_speed, action_space)
                        print("当前电车信息：", source_roadsegment, E_idx_0, deadline_idx_0)
                        print("发电站序号：", nps_idx)
                        print("电车序号：", ev_idx)
                        print("电车起点到发电站的路径，剩余电量，剩余截止时间", path_s_nps, E_s_nps, deadline_s_nps)
                        if len(path_s_nps) > 0:
                            nps_idex_roadsegment = action_space.index(path_s_nps[len(path_s_nps) - 1])  # 发电站所在路段编号
                            E_nps_L_0 = E_s_nps + newpower_value[nps_idx]
                            deadline_nps_L_0 = deadline_s_nps
                            print("当前电车信息：", nps_idex_roadsegment, E_nps_L_0, deadline_nps_L_0)
                            path_nps_L, E_nps_L, deadline_nps_L = run_ev(nps_idex_roadsegment, 0, E_nps_L_0, deadline_nps_L_0,
                                                                         graph, graph_speed, action_space)

                            print("发电站到充电站L的路径，剩余电量，剩余截止时间", path_nps_L, E_nps_L, deadline_nps_L)
                        if len(path_s_nps) > 0 and len(path_nps_L) > 0:
                            completed_path = path_s_nps + path_nps_L
                            print("电车经过发电站到达充电站的完整路径：", completed_path)
                            if E_nps_L > max_E:
                                max_path = completed_path
                                max_E = E_nps_L
                        ev_idx = ev_idx + 1
                    if max_E > 0:
                        length_path = compute_length(graph, max_path)
                        total_cost_not_e = total_cost_not_e + gamma * length_path / newpower_value[nps_idx]
                    nps_idx = nps_idx + 1
                total_p_e = total_p_e + abs(total_cost-total_cost_not_e)+gamma*max_ev_length[ev_pe]*lbd_s
                ev_pe= ev_pe + 1
            print("total_payment_cost_ratio,total_p_e,total_cost=", total_payment_cost_ratio,total_p_e,total_cost)
            if total_cost>0 and total_p_e > 0:
                total_payment_cost_ratio = total_payment_cost_ratio + total_p_e / total_cost
            counter = counter + 1

        #统计counter次实验的均值
        number_winners = total_winners/(counter-1)
        social_efficiency = total_social_effiency/(counter-1)
        payment_cost_ratio = total_payment_cost_ratio / (counter-1)
        end_time = time.perf_counter()
        runing_time = (end_time-start_time)/(counter-1)
        data = {
            "number_of_evs":number_evs,
            "number_of_winners":number_winners,
            "social_efficiency":social_efficiency,
            "payment_cost_ratio":payment_cost_ratio,
            "running time": runing_time
        }
        save_to_meus_json(data,'meus_5.json')
        number_evs = number_evs + 1