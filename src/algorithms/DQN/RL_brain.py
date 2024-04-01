import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
np.random.seed(1)

#tf.random.set_seed(1)
# tf.set_random_seed(1)
class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False
                 ):
        self.n_actions = n_actions#可用路段，需要根据已经完成的动作更新该集合
        self.n_features = n_features#状态的特征值：当前路段，剩余电量，剩余deadline
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy # epsilon 的最大值
        self.replace_target_iter = replace_target_iter # 更换 target_net 的步数
        self.memory_size = memory_size # 记忆上限
        self.batch_size = batch_size # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy_increment# epsilon 的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self._biuld_net()# 创建 [target_net, evaluate_net]
        # print("\n [RL_brain]__init__")
        # 替换 target net 的参数
        t_params = tf.get_collection('targe_net_params')
        e_params = tf.get_collection('eval_net_params')
        # print("[RL_brain].init,t_params,e_params=",t_params,e_params)
        self.replace_target_op = [tf.assign(t,e) for t, e in zip(t_params, e_params)]
        self.sess = tf.Session()
        # 输出 tensorboard 文件
        # if output_graph:
        # $ tensorboard --logdir=logs
        # tf.summary.FileWriter("logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())#基于NN初始化Q值
        self.cost_his = []# 记录所有 cost 变化, 用于最后 plot 出来观看

    def store_transition(self,s,a,r,s_):
        # print("[RL_brain].store_transition", s, a, r, s_)
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s,[a,r],s_))# 记录一条 [s, a, r, s_] 记录
        index = self.memory_counter % self.memory_size
        # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        self.memory[index, :] = transition # 替换过程
        self.memory_counter += 1

    def choose_action(self,obervation,action_space,path_id):
        # 统一 observation 的 shape (1, size_of_observation)
        obervation=np.array(obervation)
        seg_end = action_space[int(obervation[0])][1]#当前路段的终点
        obervation = obervation[np.newaxis,:]
        # print("转换的observation：",obervation,obervation.shape)
        epp = np.random.uniform()
        # print("[RL_brain]", epp,self.epsilon)
        visited_path = []
        for idd in path_id:
            visited_path.append(action_space[idd])
        visited_path = np.array(visited_path).flatten()#二维列表转换为一维列表
        vp = set(visited_path)#路径访问过的点集合
        # print("[RL_brain].choose_action,visited_path, vp", visited_path, vp)
        if epp  < self.epsilon:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.sess.run(self.q_eval, feed_dict= {self.s: obervation})
            #神经网络计算得到的所有路段的Q值
            # print("[RL_brain].choose_action, 神经网络计算得到的actions_value=", actions_value)
            # print("[RL_brain].choose_action, type(actions_value),shape(actions_value)=",
            #       type(actions_value),actions_value.shape)
            # print("[RL_brain].choose_action,path_id=",path_id)
            for aa in path_id:#删除走过的路段及其回头路
                # print("[RL_brain].choose_action, 正在删除走过的路段 aa=",aa)
                actions_value[0][aa] = '-inf'
                # idx = action_space.index(action_space[aa])
                # actions_value[0][idx] = '-inf'
            # print("[RL_brain].choose_action, 删除走过的路段,更新后的actions_value=", actions_value)
            if vp is not None:
                # print("[RL_brain].choose_action,visited_path, vp", visited_path, vp)
                for ii in vp:
                    # print("[RL_brain].choose_action,seg_end, ii", seg_end,ii)
                    if (seg_end,ii) in action_space:
                        idex = action_space.index((seg_end,ii))
                        # print("[RL_brain].choose_action,idex", idex)
                        actions_value[0][idex] = '-inf'
            #筛选以当前路段终点为起点的路段
            for bb in range(len(action_space)):
                # print("[RL_brain].choose_action,bb, action_space[bb],seg_end=",bb, action_space[bb],seg_end)
                if action_space[bb][0] != seg_end:#该路段与当前路段不连接
                    # print("[RL_brain].choose_action,actions_value[0][bb]", actions_value[0][bb])
                    actions_value[0][bb] = '-inf'
            # print("[RL_brain].choose_action, 删除不相连接路段，更新后的actions_value=", actions_value)
            action = np.argmax(actions_value)
        else:
            # 随机选择
            # flag = True#随机生成一个路段编号，判断它是否与当前路段相连接，且没有访问过，否则继续循环
            # action = 'inf'
            # while flag:
            #     action_tmp = np.random.randint(0, self.n_actions)
            # print("[RL_brain] action_tmp, flag, action_space[action_tmp],seg_end,vp ",
            #       action_tmp, flag, action_space[action_tmp],seg_end,vp)
            # if (action_space[action_tmp][0] == seg_end
            #         and (action_space[action_tmp][0] in vp) == False
            #         and (action_space[action_tmp][1] in vp) == False):
            #             #统计某个元素在列表中出现的次数
            #     action = action_tmp
            #     flag = False
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # print("[RL_brain].learn: learn_step_counter, replace_target_iter",
        #       self.learn_step_counter,self.replace_target_iter)
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 检查是否替换 target_net 参数
            self.sess.run(self.replace_target_op)
            # print('\n [RL_brain] target_params_replaced\n')
        if self.memory_counter > self.memory_size:
            # 从 memory 中随机抽取 batch_size 这么多记忆
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory = self.memory[sample_index,:]
        # print("[RL_brain] batch_memory的后3个值=", batch_memory[:,-self.n_features:])
        # print("[RL_brain] batch_memory的前3个值=", batch_memory[:,:self.n_features])
        q_next, q_eval = self.sess.run(
            [self.q_next,self.q_eval],
            feed_dict={
                self.s_:batch_memory[:,-self.n_features:],#选择后n_features（3个）观测值作为：下一个状态
                self.s:batch_memory[:,:self.n_features],#选择前n_features（3个）观测值作为：当前状态
            })# 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        # print("[RL_brain] q_next, q_eval=",q_next,q_eval)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size,dtype=np.int32)
        eval_act_index = batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next,axis=1)
        _, self.cost = self.sess.run([self._train_op,self.loss],
                                     feed_dict={
                                         self.s:batch_memory[:,:self.n_features],
                                         self.q_target:q_target
                                     }
                                     )# 训练 eval_net
        #因为q_target是选择状态下q值最大的action，而q_eval是固定的，所以需要一些处理，便于求二者的error。
        #上述操作对矩阵运算进行了一些处理，目的是为了方便反向传递。
        #对Q估计的值进行反向传递，基于Qtarget的max值，更新Q估计
        #Qtarget不需要传递
        self.cost_his.append(self.cost)# 记录 cost 误差
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # def plot_cost(self):
    #     import matplotlib.pyplot as plt
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')
    #     plt.show()
    #
    #
    def _biuld_net(self):
        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        tf.disable_eager_execution()
        tf.reset_default_graph()
        self.s = tf.placeholder(tf.float32,[None, self.n_features], name='s')
        # print("[RL_brain]_biuld_net, type(self.s)=",type(self.s))
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name = 'Q_target')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                    tf.random_normal_initializer(0.,0.3), tf.constant_initializer(0.1)
            # eval_net 的第一层. collections 是在更新 target_net 参数时会用到
            # print("[RL_brain]w_initializer,b_initializer=", w_initializer,b_initializer)
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer = w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1)+b1)
                # print("[RL_brain]w1,b1,l1=",w1,b1,l1)
                # relu称为线性整流函数(修正线性单元),tf.nn.relu()用于将输入小于0的值增幅为0,输入大于0的值不变
                #tf.matmul是矩阵的乘法,即tf.matmul(x,y)中的x和y要满足矩阵的乘法规则
            # eval_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer,
                                     collections=c_names)
                self.q_eval = tf.matmul(l1, w2)+b2

        with tf.variable_scope('loss'):# 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):#梯度下降
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            #tf.train.rmsprop() 函数用于创建使用 RMSProp 梯度下降算法
        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        # 接收下个 observation
        with tf.variable_scope('target_net'):
            c_names=['traget_net_parmas',tf.GraphKeys.GLOBAL_VARIABLES]
            # target_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,collections=c_names)
                b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)
            # target_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

        #保存的数据是s,a,r(s,a),s'，
        #其中：s[上一个时间步骤的剩余电量(E_e^a(t-1))，上一个时间步骤的剩余deadline(t_e(t-1))]，即走完路段a的前一条路段的相关信息
        #a[s状态下选中的一条路段编号]，
        #r(s,a)[根据公式计算的一个实数]，s'[剩余电量(E_e^a(t))，剩余deadline(t_e(t))]
        #observation, observation_两个状态值分别由（行，列）于是需要两个值，
        #另外再加上reward和action，共6个参数值
        #延伸到EV调度：observation=（经度，纬度），reward=[0, 剩余电量*电价, -（耗电量*电价）]
        #action：某条路段（给所有路段编号）