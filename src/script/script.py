import requests
from bs4 import BeautifulSoup

'''
我们使用的是 `https://8684.cn/ `这个网站
我们是以常熟为例，爬取常熟的公交线路数据，需要修改以下地方：
常熟市的公交数据: https://changshu.8684.cn/

注意，我们的是不删除文件的，会直接在文件后面添加，如果没有删除文件，直接运行程序，会导致数据一直添加，会出现重复的数据

请注意，网站很有可能会更新，所以如果有界面变动，请修改标签的类名


本段代码需要修改以下的代码:
1. city = '' # 此处修改城市
2. type = '' # 此处修改需要爬取的公交线路的类型

'''

# 请确保你的城市名是对的，建议先在网站上搜索一下，看看网站上的城市名是什么
city = 'changshu' # 此处修改城市
url = 'https://' + city + '.8684.cn'

# 这里按照你的需求修改，我们是爬取 `以数字开头` 的公交路线
type = '以数字开头'


# 进入界面一


# 发起 HTTP 请求，获取城市数据
response = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
# 打印请求的状态码，如果返回200，则证明请求成功
if (response.status_code == 200):
    print("连接成功")

#获取数据并解析
soup = BeautifulSoup(response.text, 'lxml')
# 找到 `div` 标签的类名为 `bus-layer depth w120` 的元素
soup_buslayer = soup.find('div', class_='bus-layer depth w120')

# 解析分类数据
# 声明一个字典，用于存储公交线路分类信息
dic_result = {}
# 找到 `div` 标签的类名为 `pl10` 的元素
soup_buslist = soup_buslayer.find_all('div', class_='pl10')
# 遍历所有的 `div` 标签的类名为 `pl10` 的元素
for soup_bus in soup_buslist:
    # 获取分类的名称
    name = soup_bus.find('span', class_='kt').get_text()

    #我们只需要爬取 `以数字开头` 的公交路线
    if type in name:
        # 找到 `div` 标签的类名为 `list` 的元素
        soup_a_list = soup_bus.find('div', class_='list')
        # 获取list中的所有a标签
        for soup_a in soup_a_list.find_all('a'):
            # 获取链接名称
            text = soup_a.get_text()
            # 获取链接地址
            href = soup_a.get('href')
            # 将线路名称和 URL 存储到字典中
            dic_result[text] = url + href #此处需要修改

            
#
# 测试语句，判断上面的代码是否正确
# print(dic_result)
#
#


# 进入界面二


#获取公交站点地址        
# 存储公交站点地址的列表
bus_arr = []
# 记录当前遍历到的公交车站点的索引
index = 0
# 
start_num = 0
bus_arr_start_num = 0

# 遍历公交站点列表
for key,value in dic_result.items():
    # 发送 HTTP 请求，获取公交线路
    response = requests.get(url=value, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
    #获取数据并解析
    soup = BeautifulSoup(response.text, 'lxml')
    # 详细线路
    # 找到 `div` 标签的类名为 `list clearfix` 的元素
    soup_buslist = soup.find('div', class_='list clearfix')
    # 判断是否为空，因为有的城市没有以数字开头的公交车
    if soup_buslist != None:
        # 找到 `list clearfix` 标签名为 `a` 的元素
        soup_list2=soup_buslist.find_all('a')
        # 遍历当前界面所有的公交车路线
        for soup_a in soup_list2:
            # 获取 `a` 标签的名称
            text = soup_a.get_text()
            # 获取 `a` 标签的链接
            href = soup_a.get('href')
            # 获取 `a` 标签的title
            title = soup_a.get('title')
            # 储存到列表中
            bus_arr.append([title, text, url + href])
            
            
        #
        # 测试语句，判断上面的代码是否正确
        # print(bus_arr)
        #
        #


    # 进入界面三


    # 创建一个空集合来存储公交站点名称
    s = set()

    # 打开文件
    with open("data/" + city + '.txt', 'ab') as f:
        # 指向文件最后，即在文件最后插入
        f.seek(0, 2)  # set file pointer to end of file

        # 遍历公交站点的 URL
        for i in range(bus_arr_start_num, len(bus_arr)):
            bus_arr_start_num = len(bus_arr)
            # 输出进度条
            rate = int(float(i) / len(bus_arr) * 100)
            print("\r进度: {}% [{}{}]".format(rate, '*' * int(rate / 10), ' ' * (10 - int(rate / 10))), end="")

            # 获取公交车站点的 URL
            url_busstop = bus_arr[i][2]
            # 将公交车的编码设置成 `utf-8`
            key_str = bus_arr[i][1].encode("utf-8")
            # 发送 HTTP 请求，获得公交车的具体站点
            response = requests.get(url=url_busstop, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
            # 获取数据并解析
            soup = BeautifulSoup(response.text, 'lxml')
            # 找到 `div` 标签的类名为 `bus-lzlist mb15` 的元素
            soup_buslayer1 = soup.find('div', class_='bus-lzlist mb15')
            # 判断有没有这个标签
            if soup_buslayer1 != None:
                # 读取里面的所有ol
                for a in soup_buslayer1.find_all('ol'):
                    # 读取里面的所有li
                    for busstop in a.find_all('li'):
                        # 读取里面的所有 `a` 标签
                        busstopfinal = busstop.find('a')
                        # 判断是否有这个标签
                        if busstopfinal != None:
                            # 获取站台名
                            text = busstopfinal.get('aria-label')
                            # 判断是否为空
                            if text != None:
                                # 设置站台名编码为 utf-8
                                str = text.split()[1].encode("utf-8")
                                if str not in s:
                                    s.add(str)
                                    # 将公交线路信心写入文件
                                    f.write(key_str + b' ' + str + b'\n')
                                    start_num += 1
    f.close()
    print("以数字" + key + "开头的公交车数据爬取完成\n")
print("所有的数据爬取完成")


