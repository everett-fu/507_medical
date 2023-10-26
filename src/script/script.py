import requests
from bs4 import BeautifulSoup
import json
import xlwt
# from Coordin_transformlat import gcj02towgs84

'''
我们使用的是 `https://8684.cn/ `这个网站
我们是以常熟为例，爬取常熟的公交线路数据，需要修改以下地方：
常熟市的公交数据: https://changshu.8684.cn/

请注意，网站很有可能会更新，所以如果有界面变动，请修改标签的类名

1. city = '' # 此处修改城市
2. type = '' # 此处修改需要爬取的公交线路的类型

'''

# 请确保你的城市名是对的，建议先在网站上搜索一下，看看网站上的城市名是什么
city = 'changshu' # 此处修改城市
url = 'https://' + city + '.8684.cn/'

# 这里按照你的需求修改，我们是爬取 `以数字开头` 的公交路线
type = '以数字开头'

# 发起 HTTP 请求，获取城市数据
response = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'})
# 打印请求的状态码，如果返回200，则证明请求成功
print(response. status_code)

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
            dic_result[text] = city + href #此处需要修改

            
#
# 测试语句，判断上面的代码是否正确
# print(dic_result)
#
#