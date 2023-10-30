import json
import requests
import xlwt
import coordinateTransformation

'''
调用高德的API接口，查询公交车站点经纬度，以下为高德API接口的使用说明：
https://lbs.amap.com/api/webservice/guide/api/georegeo

需要修改的参数有：
city：城市名
key：高德开放平台的key
adcode：城市的adcode，可以在这个网站上查看你的城市的adcode: https://lbs.amap.com/api/webservice/download

注意，key密钥请自己申请，申请web服务类型的key

'''


# 此处修改城市
city = 'changshu'
#填入高德key
key= 'e9d2e16a0caabd242de9047405c639f2'            
# 此处修改城市的adcode，这个可以让高德只查看这个城市的公交站点，不然会查看全国的公交站点
# 你可以在这个网站上查看你的城市的adcode: https://lbs.amap.com/api/webservice/download
adcode = '320581'


#获取站点经纬度信息
car_list = []

# 高德查询经纬度的api地址
url = 'https://restapi.amap.com/v3/geocode/geo'
# 读取之前爬取的公交车站台数据，存储在car_list列表中
with open("data/" + city + '.txt', encoding='utf-8') as file_obj:
    for line in file_obj:
            car_list.append(line)

 
# 创建一个 Excel 工作簿，并在其中创建一个名为 `site` 的表格。
wb=xlwt.Workbook()
ws=wb.add_sheet('site')
# 公交车线路
ws.write(0,0,'name')
# 站台地址
ws.write(0,1,'address')
# 经度
ws.write(0,2,'longitude')
# 维度
ws.write(0,3,'latitude')

# 读取car_list列表中的公交车站点信息，并调用高德API接口，查询公交车站点经纬度
# 注意，调用高德的api有时候会被高德掐掉，但之前爬取的数据都会存在data/文件夹下，所以不用担心
# 但高德每人每天只有5000的免费额度，所以当被掐掉的时候，请先另存为数据文本，然后
# 修改下面的for循环的range的范围，重新运行程序，将几个文件拼接起来即可。
for i in range(0,len(car_list)):
    # 输出进度条
    rate = int(float(i) / len(car_list) * 100)
    print("\r进度: {}% [{}{}]".format(rate, '*' * int(rate / 10), ' ' * (10 - int(rate / 10))), end="")

    # 创建一个空列表，用于存储公交车站点信息
    f_bus_stop=[]
    # 将公交车站点信息分割成列表
    parts = car_list[i].split()
    # 公交车线路
    first_name = parts[0]
    # 公交车站台
    last_name = parts[1]
    address=last_name+"公交站"

    # 调用高德API接口，查询公交车站点经纬度
    params = { 'key': key,
           'address': address,'city':adcode } 
    res = requests.get(url, params)
    jd = json.loads(res.text)
    # 这几行代码是用于判断站点是否有经纬度信息，如果有，则将经纬度信息添加到列表f_bus_stop中。
    if 'geocodes' in jd and len(jd['geocodes']) > 0:
        coords = [float(x) for x in jd['geocodes'][0]['location'].split(',')]
    f_bus_stop.append(first_name)
    f_bus_stop.append(last_name)
 
    # 这里是坐标系转换
    # 因为高德地图给的是gcj02坐标系，而我们需要的是wgs84坐标系
    # 这里采用的是网上开源的代码，具体请看coordinateTransformation.py文件
    result = coordinateTransformation.gcj02_wgs84(float(coords[0]), float(coords[1]))
    f_bus_stop.append(result[0])
    f_bus_stop.append(result[1])

    # 将站点信息写入Excel表格中
    for j in range(0,4,1):
        ws.write(i+1, j,f_bus_stop[j])
        wb.save('data/' + city + '.xls')