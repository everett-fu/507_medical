import matplotlib.pyplot as plt
# 原始数据
data = [
    [222.93886451315598, 470.414877257099, 677.7641505896131], [245.69478229166106, 476.96009103381886, 728.9961897252907], [264.52942271885775, 466.30571498531816, 704.5376136900807], [246.56069829232456, 494.0001569402824, 720.5174820415277], [235.22789987421237, 516.8016726259094, 768.4963271409908], [243.36853098367374, 488.8171156795477, 760.5765755975282]
]

# 除数列表
divisors = [5, 10, 15]

# 处理数据
result = []
for sublist in data:
    # 使用列表推导式和enumerate同时获取元素和其索引
    # 使用索引从divisors获取相应的除数
    result.append([value / divisors[i % len(divisors)] for i, value in enumerate(sublist)])


result = data

# 转置数据以便每一列变成一个独立的列表
# 这样 data_transposed[0] 将包含所有行的第一列的数据，依此类推
data_transposed = list(zip(*result))

# 创建x轴的数据点，假设每个数据点之间的间隔是固定的
x_points = range(5, 11)

# 绘制每一列数据
plt.plot(x_points, data_transposed[0], label='5')
plt.plot(x_points, data_transposed[1], label='10')
plt.plot(x_points, data_transposed[2], label='15')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Data Columns Plot')
plt.xlabel('Data Point')
plt.ylabel('Value')

# 显示图表
plt.show()
#
# # 打印结果
# for r in result:
#     print(r)
