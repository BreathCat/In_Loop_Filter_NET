#！/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Set

import requests

"""
通过requests可以向某个地址发送请求
 # /usr/bin/python
"""

"""
response = requests.get('http://10.10.43.3/drcom/login')
# 通过get请求返回的文本值
print(response.text)
"""


# post发送的数据
postData: Set[str] = {
    'callback': 0,
    'DDDDD': 8201,
    'upass': 821119,
    '0MKKey': 123456,
    'R1': 0,
    'R3': 0,
    'R6': 0,
    'para': 00,
    'v6ip': 1574044489427,
}

# 对于我们工作中的自己人,我们一般会使用别的验证,而不是csrf_token验证
response = requests.post('http://10.10.43.3/drcom/login',data=postData)
# 通过get请求返回的文本值
print(response.text)
