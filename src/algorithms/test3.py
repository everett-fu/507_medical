import numpy as np
import tensorflow as tf
import random
from collections import deque


# 环境类
class Environment:
    # 初始化环境
    def __init__(self, adjacency_matrix, charge_matrix, speed_matrix, start, end, timelimit, energy, energy_consume, charge_power):
        self.adjacency_matrix = adjacency_matrix    # 距离邻接矩阵
        self.charge_matrix = charge_matrix  # 充电邻接矩阵
        self.speed_matrix = speed_matrix    # 速度邻接矩阵
        self.start = start  # 起始节点
        self.end = end  # 结束节点
        self.timelimit = timelimit  # 时间限制
        self.energy = energy    # 电量限制
        self.energy_consume = energy_consume    # 耗电量，每公里多少电
        self.charge_power = charge_power    # 充电功率，每小时多少电

        self.current_state = start  # 将当前节点初始化为起始节点
        self.time = timelimit   # 初始化时间
        self.power = energy     # 初始化电量
        self.path = [start]     # 初始化访问过节点

    # 重置环境
    def reset(self):
        self.current_state = self.start     # 设置当前节点初始化为起始节点
        self.time = self.timelimit  # 重置时间限制
        self.power = self.energy    # 重置电量限制
        self.path = [self.start]    # 重置访问过的节点
        return self.current_state

    # 执行动作
    def step(self, action):
        next_node = action
        use_time = self.adjacency_matrix[self.current_state, next_node] / self.speed_matrix[self.current_state, next_node]    # 选择这条路的耗时
        done = False
        time_ = self.time
        power_ = self.power

        if next_node in self.path:
            reward = -100  # 如果节点已经访问过，给予大的负奖励
            next_node = self.current_state  # 保持当前位置不变
        elif self.adjacency_matrix[self.current_state, next_node] == 0:
            reward = -100  # 如果尝试非法移动（即两个节点间没有直接连接），同样给予大的负奖励
            next_node = self.current_state  # 保持当前位置不变
        else:
            self.path.append(next_node)  # 如果是有效且未访问过的节点，加入访问集合
            # print(f"adjacency_matrix: {self.adjacency_matrix[self.current_state, next_node]}")
            charge = -self.adjacency_matrix[self.current_state, next_node] * self.energy_consume     # 这次电量的增加或减少的值
            # print(f"charge_1: {charge}")
            # 如果是充电路段
            if self.charge_matrix[self.current_state, next_node] == 1:
                # print(f"use_time: {use_time}")
                # print(f"power: {use_time * self.charge_power}")
                charge += use_time * self.charge_power
                # print(f"charge_2: {charge}")
            power_ += charge
            time_ -= use_time
            # self.power += charge    # 加上电量的变化值
            # self.time -= use_time   # 减去消耗时间
            reward = charge
            # 如果时间或者电量不满足
            if time_ < 0 or power_ < 0:
                next_node = self.current_state  # 保持当前位置不变
                reward = -10
                done = True
            elif time_ > 0 and power_:
                self.power = power_
                self.time = time_

        if not done:
            self.current_state = next_node   # 更新节点为下一个节点
            done = self.current_state == self.end    # 检查是否到达目标节点
        return next_node, reward, done

        # reward = self.adjacency_matrix[self.current_state, action]
        # self.current_state = action
        # self.path.append(action)  # Add action to path
        # done = self.current_state == self.end
        # return self.current_state, reward, done

    # 获取路径
    def get_path(self):
        return self.path

    # 获取剩余电量
    def get_energy(self):
        return self.power


# DQN代理
class DQN:
    def __init__(self, state_size, action_size):
        # 初始化DQN模型的参数
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)    # 经验回放的记忆队列
        self.gamma = 0.95   # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01     # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减值
        self.learning_rate = 0.001  # 学习率
        self.model = self._build_model()    # 创建DQN网络模型

    def _build_model(self):
        # 创建深度Q网络模型

        # 输入层：接受当前状态作为输入
        # 'None' 在这里意味着批量大小可以是任意的，self.state_size 是输入的维度
        inputs = tf.placeholder(tf.float32, [None, self.state_size])
        # 动作：这是一个与预测的Q值相乘的向量，用于选择执行的动作
        actions = tf.placeholder(tf.float32, [None, self.action_size])
        # 目标：这是我们希望Q值学习达到的目标
        targets = tf.placeholder(tf.float32, [None])

        # 网络层
        # 第一层全连接层：输入层连接到一个有24个神经元和ReLU激活函数的隐藏层
        fc1 = tf.layers.dense(inputs, 24, activation=tf.nn.relu)
        # 第二层全连接层：第一隐藏层连接到另一个有24个神经元和ReLU激活函数的隐藏层
        fc2 = tf.layers.dense(fc1, 24, activation=tf.nn.relu)
        # 输出层：最后一个隐藏层连接到输出层，输出层的大小等于动作空间的大小，没有激活函数，直接输出预测的Q值
        predictions = tf.layers.dense(fc2, self.action_size)

        # 计算损失：首先，通过actions和predictions的点乘得到实际采取动作的Q值。
        # 然后，计算这个Q值与目标Q值的均方误差，这是我们要通过训练最小化的损失。
        action_values = tf.reduce_sum(tf.multiply(predictions, actions), axis=1)
        loss = tf.reduce_mean(tf.square(targets - action_values))

        # 优化器：使用Adam优化器来最小化损失，通过调整网络参数来减少目标和预测之间的误差
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # 返回构建的网络组件，包括输入、动作、目标占位符，预测输出，优化器和损失
        return inputs, actions, targets, predictions, optimizer, loss

    # 保存记忆
    def remember(self, state, action, reward, next_state, done):
        # 将经验（状态，动作，奖励，下一个状态，是否结束）添加到记忆中
        self.memory.append((state, action, reward, next_state, done))

    # 选择动作
    def act(self, state, sess):
        # 以epsilon的概率随机选择一个动作（探索）
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # 否则，利用当前网络模型预测每个动作的Q值，并选择Q值最高的动作
        act_values = sess.run(self.model[3], feed_dict={self.model[0]: state})
        return np.argmax(act_values[0])

    # 经验回放，训练网络
    def replay(self, batch_size, sess):
        # 从记忆中随机抽取一批经验
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # 计算目标Q值：如果next_state是最终状态，则目标Q值为reward；否则，目标Q值是reward加上折扣后的未来最大Q值
            target = reward if done else reward + self.gamma * np.amax(
                sess.run(self.model[3], feed_dict={self.model[0]: next_state})[0])
            # 获取当前状态下所有动作的预测Q值
            target_f = sess.run(self.model[3], feed_dict={self.model[0]: state})
            # 更新执行的动作对应的Q值为我们计算的目标Q值
            target_f[0][action] = target
            # 使用这个更新后的Q值数组进行训练，以此来更新网络的权重
            sess.run([self.model[4]],
                     feed_dict={self.model[0]: state, self.model[2]: [target], self.model[1]: target_f})
        # 如果epsilon大于最小值，则按照衰减率减少epsilon，减少随机行为，增加根据模型预测的行为
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 训练DQN主函数
def train_dqn(episodes):
    adjacency_matrix = np.array([
        [0., 69.06484223, 75.57585925, 71.65824584, 66.8282913],
        [69.06484223, 0., 7.68940613, 12.99483813, 2.99111645],
        [75.57585925, 7.68940613, 0., 17.89846051, 8.96155748],
        [71.65824584, 12.99483813, 17.89846051, 0., 15.32024517],
        [66.8282913, 2.99111645, 8.96155748, 15.32024517, 0.]
    ])
    charge_matrix = np.array([
        [0., 1., 0., 1., 0.],
        [1., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
    speed_matrix = np.array([
        [11, 42, 25, 32, 41],
        [50, 58, 22, 21, 22],
        [52, 18, 57, 37, 59],
        [39, 53, 54, 60, 25],
        [57, 19, 10, 37, 48]])
    start_node = 0
    goal_node = 4
    timelimit = 4
    energy = 13.4
    energy_consume = 0.1344
    charge_power = 200

    # 创建环境
    env = Environment(
        adjacency_matrix=adjacency_matrix,
        charge_matrix=charge_matrix,
        speed_matrix=speed_matrix,
        start=start_node,
        end=goal_node,
        timelimit=timelimit,
        energy=energy,
        energy_consume=energy_consume,
        charge_power=charge_power)
    # 获取状态的大小与动作的大小，这里都是节点的数量
    state_size = adjacency_matrix.shape[0]
    action_size = adjacency_matrix.shape[0]
    agent = DQN(state_size, action_size)

    # 开启一个TensorFlow会话
    with tf.Session() as sess:
        # 初始化所有的全局变量
        sess.run(tf.global_variables_initializer())

        # 对于每一个episode
        for e in range(episodes):
            # 重置环境， 获取初始化状态
            state = env.reset()
            # 对状态进行one-hot编码，为了与网络的输入匹配
            state = np.identity(state_size)[state:state + 1]  # one hot encoding

            # 在一个episode中，最多执行500个时间步
            for time in range(500):
                # 根据当前状态选择有一个动作
                action = agent.act(state, sess)
                # 执行动作，观察下一个状态、奖励和是否结束
                next_state, reward, done = env.step(action)
                # 对下一个状态进行one-hot编码
                next_state = np.identity(state_size)[next_state:next_state + 1]  # one hot encoding
                # 将这个装换（状态，动作，奖励，下一个状态，是否结束）保存到记忆中
                agent.remember(state, action, reward, next_state, done)
                # 更新当前状态为下一个状态
                state = next_state
                # 如果达到终止状态
                if done:
                    print(f"Episode: {e + 1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                    print(f"Path: {env.get_path()}, Energy: {env.get_energy()}")
                    break
                # 如果记忆中的经验足够，开始训练网络
                if len(agent.memory) > 32:
                    agent.replay(32, sess)
            # 如果epsilon（探索率）大于最小值，则对其进行衰减，以减少探索行为，增加利用学到的策略
            # if agent.epsilon > agent.epsilon_min:
                # agent.epsilon *= agent.epsilon_decay


if __name__ == "__main__":
    train_dqn(500)
