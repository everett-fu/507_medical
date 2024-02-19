from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    # 初始化环境
    step = 0
    # 一共训练300次
    for episode in range(300):
        # 每次训练都初始化环境
        observation = env.reset()

        while True:
            # 更新环境
            env.render()

            # 使用DQN根据当前观测值选择一个动作
            action = RL.choose_action(observation)

            # 根据这次的动作，获取选择这个动作以后的观测值，奖励，并判断游戏是否结束
            observation_, reward, done = env.step(action)

            # 根据上面获得的观察值，所选择的动作，奖励，下一次的观测值存储到DQN表中
            RL.store_transition(observation, action, reward, observation_)

            # 如果步数超过200步，并且为5步数的倍数进行训练
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 更新观察值
            observation = observation_

            # 当结束的时候，跳出循环
            if done:
                break
            step += 1

    # 结束游戏，并销毁环境
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # 创建环境
    env = Maze()
    # 创建DQN网络
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    # 100毫秒后调用run_maze函数
    env.after(100, run_maze)
    # 使用Tkinter的主循环
    env.mainloop()
    # 可视化学习成果
    RL.plot_cost()