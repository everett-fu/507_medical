import pandas as pd
import numpy as np

# 从 Excel 文件中读取数据
file_path = 'data/changshu.xls'
data = pd.read_excel(file_path)

# 根据需要选择指定数量的随机点
def select_random_points(data, n):
    # 从数据中随机选择 n 个点
    selected_points = data.sample(n)
    return selected_points

# 构建邻接矩阵函数
def build_adjacency_matrix(data, size):
    # 创建一个空的邻接矩阵
    adjacency_matrix_distance = np.zeros((size, size))  # 距离邻接矩阵
    adjacency_matrix_charge = np.zeros((size, size))  # 充电路段邻接矩阵
    adjacency_matrix_speed = np.zeros((size, size))  # 速度邻接矩阵

    # TODO: 在这里实现根据数据构建邻接矩阵的逻辑

    return adjacency_matrix_distance, adjacency_matrix_charge, adjacency_matrix_speed

# 主函数
def main(random_speed, random_points, n, m):
    # 读取数据并选择指定数量的随机点
    if random_points == -1:
        selected_points = select_random_points(data, n)
    else:
        # 这里需要根据特定要求选择点
        pass

    # 构建邻接矩阵
    size = len(selected_points)
    distance_matrix, charge_matrix, speed_matrix = build_adjacency_matrix(selected_points, size)

    # 生成随机速度矩阵（如果需要）
    if random_speed == -1:
        # 生成随机速度矩阵的逻辑
        pass

    return distance_matrix, charge_matrix, speed_matrix

# 测试代码
random_speed = -1  # 是否随机速度
random_points = -1  # 是否随机选取点
n = 10  # 选取的随机点数量
m = 5  # 选一条 m 个点的路径作为充电路段
size = 10  # n*n 的矩阵
distance_matrix, charge_matrix, speed_matrix = main(random_speed, random_points, n, m)
print(distance_matrix, charge_matrix, speed_matrix)

