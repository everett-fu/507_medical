import random
import pandas as pd
import numpy as np
import math


# 随机选取n行的数据，返回n行dataframe数据
def select_random_points(data, n):
    # 从数据中随机选择 n 个点
    selected_points = data.sample(n)
    return selected_points


# 计算两个点之间的欧式距离，返回km
def euclidean_distance(point1, point2):
    # 赋值经纬度
    lon1, lat1 = point1
    lon2, lat2 = point2

    # 将经纬度转换为弧度
    radian = lambda x: x * math.pi / 180

    # 地球平均半径（单位：米）
    R = 6371009
    dlon = radian(lon2 - lon1)
    dlat = radian(lat2 - lat1)

    a = (math.sin(dlat / 2)) ** 2 + math.cos(radian(lat1)) * math.cos(radian(lat2)) * (math.sin(dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance / 1000


# 构建邻接矩阵函数
'''
这个函数用于生成三个邻接矩阵，分别是距离邻接矩阵，充电路段邻接矩阵，速度邻接矩阵
该函数有五个输入，分别是
    + 数据源（为爬取下来的xls数据）
    + 是否是随机速度（如果是-1则生成一个40-60km/h速度邻接矩阵，如果是一个正数，生成一个常数的邻接矩阵）
    + 是否是随机点位（如果是-1则随机从文件取n个点，m个点组成的充电路段）
    + 随机点位模式下取的n位数量             注意：这个值在启用随机点位的情况下才有用，如果没有启用随机点位，可以不用输入
    + 随机点位模式下取的m个点组成的充电路段   注意：这个值在启用随机点位的情况下才有用，如果没有启用随机点位，可以不用输入

该函数的所有输入都没有进行判断，请在输入的时候先自行判断

最后返回三个邻接矩阵，分别是距离邻接矩阵，充电路段邻接矩阵，速度邻接矩阵，单位分别是km，0/1，km/h
'''
def build_adjacency_matrix(data, random_speed, random_point, n=None, m=None):
    # 判断是不是随机生成点位
    # 是随机生成点位
    if random_point == -1:
        points = select_random_points(data, n)
        charge_points = select_random_points(points, m)
    # 不是随机生成
    else:
        #
        # 只是占位，下面没有实际作用
        #
        points = select_random_points(data, 5)
        charge_points = select_random_points(points, 3)

    # 将dataframe数据变成一个二维列表
    points = points[['longitude', 'latitude']].values.tolist()
    charge_points = charge_points[['longitude', 'latitude']].values.tolist()

    # 检测点，判断取数据是否正确
    # print(points)
    # print(charge_points)

    # 矩阵的尺寸
    size = len(points)

    # 生成距离邻接矩阵
    # 初始化矩阵
    adjacency_matrix_distance = np.zeros((size, size))
    # 遍历二维矩阵，填入计算后的值
    for i in range(size):
        for j in range(size):
            if i != j:
                adjacency_matrix_distance[i, j] = euclidean_distance(points[i], points[j])

    # 检测点，用于判断距离邻接矩阵是否正确
    # print(adjacency_matrix_distance)

    # 创建一个充电路段邻接矩阵
    adjacency_matrix_charge = np.zeros((size, size))
    # 遍历二维矩阵，填入邻接矩阵
    for i in range(size):
        for j in range(size):
            if i != j and points[i] in charge_points and points[j] in charge_points:
                adjacency_matrix_charge[i, j] = 1

    # 检测点，判断充电矩阵生成是否正确
    # print(adjacency_matrix_charge)

    # 创建一个速度矩阵
    # 如果random_speed 的值为-1，采用随机生成40-60数据
    if random_speed == -1:
        adjacency_matrix_speed = np.random.randint(40, 61, size=(size, size))  # 充电路段邻接矩阵
        adjacency_matrix_speed = np.triu(adjacency_matrix_speed) + np.triu(adjacency_matrix_speed, 1).T
    # 不是随机生成点位，采用输入的数字填充矩阵
    else:
        adjacency_matrix_speed = np.full((size, size), random_speed)  # 充电路段邻接矩阵
    # 生成对称的矩阵，对角线变为0
    np.fill_diagonal(adjacency_matrix_speed, 0)

    # 检测点，判断速度矩阵生成是否正确
    # print(adjacency_matrix_speed)

    return adjacency_matrix_distance, adjacency_matrix_charge, adjacency_matrix_speed


# 初始化电车信息
# 输入：电车数量，节点数量
'''
这个函数用于初始化电车信息
该函数有两个输入，分别是
    + 电车数量
    + 节点数量
该函数的输出是一个列表，列表中的每一个元素是一个列表。包含了电车的起始路段，目标地址，初始电量，初始deadline
'''
def ini_evs_info(number_evs, number_nodes):
    # 用于存储电车的起始路段，目标地址，初始电量，初始deadline
    evs_info = []
    for i in range(number_evs):
        # 随机生成电车的起始路段
        start = random.randint(0, number_nodes - 1)
        # 随机生成电车的目标地址，直到不等于起始地址
        end = start
        while end == start:
            end = random.randint(0, number_nodes - 1)
        # 随机生成电车的初始电量round
        power = round(random.uniform(50, 80), 2)
        # 随机生成电车的初始deadline
        deadline = round(random.uniform(1, 5), 2)
        evs_info.append([start, end, power, deadline])
    return evs_info

# 用于测试
def main(data, random_speed, random_point, n=None, m=None):
    adjacency_matrix_distance, adjacency_matrix_charge, adjacency_matrix_speed = \
        build_adjacency_matrix(data, random_speed, random_point, n, m)
    print(adjacency_matrix_distance)
    print(adjacency_matrix_charge)
    print(adjacency_matrix_speed)



if __name__ == "__main__":
    # 数据地址
    file_path = '../script/data/changshu.xls'
    # 读取数据
    data = pd.read_excel(file_path)

    main(data, -1, -1, 5, 3)
    print('\n')
    main(data, 50, 201)

    evs_info = ini_evs_info(5, 5)
    print(evs_info)
