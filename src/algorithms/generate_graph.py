import pandas as pd
import numpy as np
import math

# 从 Excel 文件中读取数据
file_path = '../script/data/changshu.xls'
data = pd.read_excel(file_path)


# 随机选取n行的数据，返回n行dataframe数据
def select_random_points(data, n):
    # 从数据中随机选择 n 个点
    selected_points = data.sample(n)
    return selected_points


# 使用勾股定理计算坐标，这个方式不太准确，球面计算的库不太常见，不太好下，先使用勾股定理，后面替换
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # 用的是经纬度计算，经纬度差一度地图上相差100km，所以下面乘100
    return distance * 100


# 构建邻接矩阵函数
'''
这个函数用于生成三个邻接矩阵，分别是距离邻接矩阵，充电路段邻接矩阵，速度邻接矩阵
该函数有七个输入，分别是
    + 数据源（为爬取下来的xls数据）
    + 每公里充电数
    + 每公里耗电数
    + 是否是随机速度（如果是-1则生成一个随机的速度邻接矩阵，如果是一个正数，生成一个常数的邻接矩阵）
    + 是否是随机点位（如果是-1则随机从文件取n个点，m个点组成的充电路段）
    + 随机点位模式下取的n位数量             注意：这个值在启用随机点位的情况下才有用，如果没有启用随机点位，可以不用输入
    + 随机点位模式下取的m个点组成的充电路段   注意：这个值在启用随机点位的情况下才有用，如果没有启用随机点位，可以不用输入

该函数的所有输入都没有进行判断，请在输入的时候先自行判断
'''


def build_adjacency_matrix(data, charge, power_consumption, random_speed, random_point, n=None, m=None):
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

    # 创建一个充电路段邻接矩阵，正数为每公里充电数，负数为每公里耗电数
    adjacency_matrix_charge = np.full((size, size), power_consumption)
    # 遍历二维矩阵，填入邻接矩阵
    for i in range(size):
        for j in range(size):
            if i != j and points[i] in charge_points and points[j] in charge_points:
                adjacency_matrix_charge[i, j] = charge


    # 检测点，判断充电矩阵生成是否正确
    # print(adjacency_matrix_charge)

    # 创建一个速度矩阵
    # 如果random_speed 的值为-1，采用随机生成40-60数据
    if random_speed == -1:
        adjacency_matrix_speed = np.random.randint(40, 61, size=(size, size))  # 充电路段邻接矩阵
    # 不是随机生成点位，采用输入的数字填充矩阵
    else:
        adjacency_matrix_speed = np.full((size, size), random_speed)  # 充电路段邻接矩阵

    # 检测点，判断速度矩阵生成是否正确
    # print(adjacency_matrix_speed)


    # 利用距离矩阵，速度矩阵，是否为充电路段，生成奖励矩阵。正数为充电量，负数为耗电量
    # 先计算时间矩阵，计算方式：距离除以速度
    adjacency_matrix_distance_np = np.array(adjacency_matrix_distance)
    adjacency_matrix_speed_np = np.array(adjacency_matrix_speed)
    adjacency_matrix_time_np = adjacency_matrix_distance_np / adjacency_matrix_speed_np

    # 计算奖励矩阵，时间矩阵乘是否是充电的矩阵，正数为充电的时间，负数为耗电的时间
    adjacency_matrix_award_np = adjacency_matrix_time_np * adjacency_matrix_charge
    adjacency_matrix_award = adjacency_matrix_award_np.tolist()
    return adjacency_matrix_award


# # 用于测试
#def main(data, charge, power_consumption, randam_speed, random_point, n=None, m=None):
#   adjacency_matrix_award = \
#       build_adjacency_matrix(data, charge, power_consumption, randam_speed, random_point, n, m)
#   print(adjacency_matrix_award)


#if __name__ == "__main__":
#    main(data, 30, -50, -1, -1, 5, 3)
#    print('\n')