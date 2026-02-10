#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
从多个数据源获取双色球历史数据
优先使用官方API，其次尝试 china-ssq.net
"""
import requests
import os
import re
import time
import json
from bs4 import BeautifulSoup
from datetime import datetime


class SSQDataFetcherChinaSSQ:
    def __init__(self, max_retries=3, retry_delay=2):
        self.base_url = "https://cp.china-ssq.net/ssq"
        # 官方API（根据搜索结果）
        self.official_api = "http://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        self.data_file = "ssq_history.csv"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }
        
    def fetch_from_official_api(self):
        """
        从官方API获取数据（最可靠的方式）
        """
        print("尝试从官方API获取数据...")
        
        try:
            params = {
                'name': 'ssq',
                'issueCount': '',
                'pageNo': '1',
                'pageSize': '9999',  # 获取所有数据
                'systemType': 'PC'
            }
            
            response = requests.get(self.official_api, params=params, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('result'):
                    results = data['result']
                    print(f"从官方API获取到 {len(results)} 条数据")
                    
                    # 打印第一条数据的结构，用于调试
                    if results:
                        print(f"数据示例（第一条）: {json.dumps(results[0], ensure_ascii=False, indent=2)}")
                    
                    parsed_data = self.parse_official_api_data(results)
                    if parsed_data:
                        print(f"成功解析 {len(parsed_data)} 条数据")
                    else:
                        print("警告: 未能解析任何数据，请检查数据格式")
                    return parsed_data
                else:
                    print("官方API返回数据格式异常")
                    print(f"返回的JSON结构: {list(data.keys())}")
            else:
                print(f"官方API请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"从官方API获取数据失败: {e}")
        
        return []
    
    def parse_official_api_data(self, results):
        """
        解析官方API返回的数据
        """
        data_list = []
        
        for item in results:
            try:
                # 期号
                period = item.get('code', '')
                if not period:
                    continue
                
                # 初始化变量
                red_balls = []
                blue_ball = 0
                
                # 方式1: 尝试从数组格式获取（redBalls 和 blueBall）
                if 'redBalls' in item:
                    red_data = item.get('redBalls', [])
                    blue_data = item.get('blueBall', '')
                    
                    if isinstance(red_data, list) and len(red_data) >= 6:
                        red_balls = [int(x) for x in red_data[:6]]
                        if isinstance(blue_data, (int, str)):
                            blue_ball = int(blue_data)
                        elif isinstance(blue_data, list) and blue_data:
                            blue_ball = int(blue_data[0])
                        
                        if red_balls and blue_ball:
                            red_balls.sort()
                            if len(set(red_balls)) == 6 and all(1 <= r <= 33 for r in red_balls) and 1 <= blue_ball <= 16:
                                data_list.append({
                                    '期号': period,
                                    '开奖日期': item.get('date', item.get('drawDate', '')),
                                    '红球1': str(red_balls[0]),
                                    '红球2': str(red_balls[1]),
                                    '红球3': str(red_balls[2]),
                                    '红球4': str(red_balls[3]),
                                    '红球5': str(red_balls[4]),
                                    '红球6': str(red_balls[5]),
                                    '蓝球': str(blue_ball)
                                })
                                continue
                
                # 方式2: 从字符串格式解析（官方API格式：red="01,03,05,18,29,32", blue="04"）
                red_str = item.get('red', '')
                blue_str = item.get('blue', '')
                
                if not red_str or not blue_str:
                    continue
                
                # 解析红球（逗号分隔的字符串，如 "01,03,05,18,29,32"）
                if ',' in red_str:
                    red_nums = [x.strip() for x in red_str.split(',')]
                else:
                    # 尝试其他分隔符
                    red_nums = re.findall(r'\d+', red_str)
                
                if len(red_nums) >= 6:
                    red_balls = [int(x) for x in red_nums[:6]]
                else:
                    continue
                
                # 解析蓝球（字符串，如 "04"）
                blue_nums = re.findall(r'\d+', blue_str)
                if blue_nums:
                    blue_ball = int(blue_nums[0])
                else:
                    continue
                
                # 验证数据
                if not red_balls or len(set(red_balls)) != 6 or any(r < 1 or r > 33 for r in red_balls):
                    continue
                if blue_ball < 1 or blue_ball > 16:
                    continue
                
                red_balls.sort()
                
                data_list.append({
                    '期号': period,
                    '开奖日期': item.get('date', item.get('drawDate', '')),
                    '红球1': str(red_balls[0]),
                    '红球2': str(red_balls[1]),
                    '红球3': str(red_balls[2]),
                    '红球4': str(red_balls[3]),
                    '红球5': str(red_balls[4]),
                    '红球6': str(red_balls[5]),
                    '蓝球': str(blue_ball)
                })
                
            except Exception as e:
                # 只打印前几个错误，避免刷屏
                if len(data_list) < 3:
                    print(f"解析数据项失败: {e}")
                continue
        
        return data_list
    
    def fetch_from_china_ssq(self):
        """
        从 china-ssq.net 网站获取数据
        需要分析网页结构或找到API接口
        """
        print("尝试从 china-ssq.net 获取数据...")
        
        try:
            # 尝试访问主页面
            response = requests.get(self.base_url, headers=self.headers, timeout=15)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return []
            
            # 查找可能的API端点
            # 检查页面中的script标签，查找API调用
            soup = BeautifulSoup(response.text, 'html.parser')
            scripts = soup.find_all('script')
            
            for script in scripts:
                if script.string:
                    # 查找API URL
                    api_match = re.search(r'["\'](https?://[^"\']*api[^"\']*)["\']', script.string)
                    if api_match:
                        api_url = api_match.group(1)
                        print(f"找到可能的API: {api_url}")
                        # 可以尝试调用这个API
            
            print("提示: 需要使用浏览器开发者工具查看实际的数据接口")
            print("步骤:")
            print("1. 打开 https://cp.china-ssq.net/ssq")
            print("2. 按F12打开开发者工具")
            print("3. 切换到 Network 标签")
            print("4. 刷新页面或点击相关链接")
            print("5. 查找返回数据的请求（通常是JSON格式）")
            print("6. 复制请求URL和参数，添加到脚本中")
            
            return []
            
        except Exception as e:
            print(f"从 china-ssq.net 获取数据失败: {e}")
            return []
    
    def write_to_csv(self, data_list):
        """
        写入CSV文件，避免重复
        """
        if not data_list:
            print("没有数据可写入")
            return
        
        # 检查文件是否存在
        file_exists = os.path.exists(self.data_file)
        
        # 读取已有数据，避免重复
        existing_periods = set()
        if file_exists:
            try:
                with open(self.data_file, 'r', encoding='utf-8-sig') as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # 跳过表头
                        parts = line.strip().split(',')
                        if parts:
                            existing_periods.add(parts[0])
            except:
                pass
        
        # 写入数据
        new_count = 0
        with open(self.data_file, 'a', encoding='utf-8-sig') as f:
            if not file_exists:
                # 写入表头
                f.write('期号,开奖日期,红球1,红球2,红球3,红球4,红球5,红球6,蓝球\n')
            
            for data in data_list:
                if data['期号'] not in existing_periods:
                    f.write(f"{data['期号']},{data['开奖日期']},{data['红球1']},{data['红球2']},{data['红球3']},{data['红球4']},{data['红球5']},{data['红球6']},{data['蓝球']}\n")
                    new_count += 1
                    existing_periods.add(data['期号'])
        
        print(f"\n成功写入 {new_count} 条新数据")
        print(f"数据已保存到: {self.data_file}")
    
    def fetch_all(self):
        """
        获取所有历史数据的主方法
        优先使用官方API
        """
        print("=" * 60)
        print("获取双色球历史数据")
        print("=" * 60)
        
        # 优先使用官方API
        results = self.fetch_from_official_api()
        
        if results:
            self.write_to_csv(results)
            print("\n✓ 使用官方API成功获取数据")
            return
        
        # 如果官方API失败，尝试其他方式
        print("\n官方API不可用，尝试其他方式...")
        results = self.fetch_from_china_ssq()
        
        if results:
            self.write_to_csv(results)
        else:
            print("\n" + "=" * 60)
            print("无法自动获取数据")
            print("=" * 60)
            print("建议:")
            print("1. 检查网络连接")
            print("2. 使用浏览器开发者工具查找实际的数据接口")
            print("3. 或者使用其他数据源（如 data_fetcher.py 或 data_fetcher_500_com.py）")
            print("=" * 60)


def main():
    fetcher = SSQDataFetcherChinaSSQ()
    fetcher.fetch_all()


if __name__ == '__main__':
    main()
