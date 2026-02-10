#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
从500.com网站获取双色球历史数据
改进版：修复期号格式，添加错误处理，与现有数据格式兼容
"""
import requests
import os
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime

class SSQDataFetcher500Com:
    def __init__(self, max_retries=3, retry_delay=2):
        # 使用官方API（更稳定可靠）
        self.official_api = "http://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        self.base_url = "http://kaijiang.500.com/ssq.shtml"  # 备用数据源
        self.data_file = "ssq_history.csv"
        self.max_retries = max_retries  # 最大重试次数
        self.retry_delay = retry_delay  # 重试延迟（秒）
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        self.failed_urls = []  # 记录失败的URL，最后重试
        
    def convert_period_format(self, period_str):
        """
        转换期号格式
        500.com返回的期号可能是5位数字（如25127），需要转换为7位格式（如2025127）
        """
        # 移除所有非数字字符
        period = re.sub(r'\D', '', str(period_str))
        
        if len(period) == 5:
            # 5位期号：前2位是年份后两位，后3位是期数
            # 例如：25127 -> 2025127
            year_suffix = period[:2]
            period_num = period[2:]
            
            # 判断年份：00-25可能是2000-2025，25-99可能是2025-2099
            if int(year_suffix) <= 25:
                year = 2000 + int(year_suffix)
            else:
                year = 1900 + int(year_suffix)
            
            return f"{year}{period_num}"
        elif len(period) == 7:
            # 已经是7位格式，直接返回
            return period
        else:
            # 其他格式，尝试修复
            print(f"  警告: 期号格式异常: {period_str} (长度: {len(period)})")
            return period
    
    def download(self, url, page_str, retry_count=0):
        """
        从单个期号详情页面下载数据（带重试机制）
        """
        try:
            # 确保URL完整
            if not url.startswith('http'):
                url = "http://kaijiang.500.com" + url
            
            # 使用Session保持连接，提高效率
            session = requests.Session()
            session.headers.update(self.headers)
            
            response = session.get(url, timeout=15)
            response.encoding = 'gb2312'
            
            # 处理503错误（服务不可用，通常是限流）
            if response.status_code == 503:
                if retry_count < self.max_retries:
                    wait_time = self.retry_delay * (retry_count + 1)  # 指数退避
                    print(f"  503错误，等待{wait_time}秒后重试 ({retry_count + 1}/{self.max_retries})...")
                    time.sleep(wait_time)
                    return self.download(url, page_str, retry_count + 1)
                else:
                    print(f"  请求失败，状态码: {response.status_code} (已重试{self.max_retries}次)")
                    return None
            
            # 处理其他非200状态码
            if response.status_code != 200:
                if retry_count < self.max_retries and response.status_code in [429, 502, 504]:
                    # 429(限流), 502(网关错误), 504(网关超时)也重试
                    wait_time = self.retry_delay * (retry_count + 1)
                    print(f"  状态码{response.status_code}，等待{wait_time}秒后重试 ({retry_count + 1}/{self.max_retries})...")
                    time.sleep(wait_time)
                    return self.download(url, page_str, retry_count + 1)
                else:
                    print(f"  请求失败，状态码: {response.status_code}")
                    return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找号码元素
            ball_list = soup.select('div.ball_box01 ul li')
            
            if len(ball_list) < 7:
                print(f"  警告: 第{page_str}期号码元素不足，期望7个，实际{len(ball_list)}个")
                return None
            
            # 提取号码
            ball = []
            for li in ball_list[:7]:  # 只取前7个
                num = li.string if li.string else li.get_text().strip()
                ball.append(num)
            
            # 转换期号格式
            period = self.convert_period_format(page_str)
            
            # 提取开奖日期
            date = self.extract_date(soup)
            
            # 验证数据
            if len(ball) == 7:
                # 前6个是红球，最后1个是蓝球
                red_balls = [int(b) for b in ball[:6]]
                blue_ball = int(ball[6])
                
                # 验证红球范围（1-33）且不重复
                if len(set(red_balls)) != 6 or any(r < 1 or r > 33 for r in red_balls):
                    print(f"  警告: 第{period}期红球数据异常: {red_balls}")
                    return None
                
                # 验证蓝球范围（1-16）
                if blue_ball < 1 or blue_ball > 16:
                    print(f"  警告: 第{period}期蓝球数据异常: {blue_ball}")
                    return None
                
                # 排序红球
                red_balls.sort()
                
                return {
                    '期号': period,
                    '开奖日期': date,
                    '红球1': str(red_balls[0]),
                    '红球2': str(red_balls[1]),
                    '红球3': str(red_balls[2]),
                    '红球4': str(red_balls[3]),
                    '红球5': str(red_balls[4]),
                    '红球6': str(red_balls[5]),
                    '蓝球': str(blue_ball)
                }
            
            return None
            
        except Exception as e:
            print(f"  下载第{page_str}期数据失败: {e}")
            return None
    
    def extract_date(self, soup):
        """
        从详情页面提取开奖日期
        """
        try:
            # 尝试多种方式提取日期
            # 方式1: 查找包含日期的元素
            date_patterns = [
                r'(\d{4}[-年]\d{1,2}[-月]\d{1,2})',
                r'(\d{4}/\d{1,2}/\d{1,2})',
                r'开奖日期[：:]\s*(\d{4}[-年]\d{1,2}[-月]\d{1,2})',
            ]
            
            text = soup.get_text()
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    date_str = match.group(1)
                    # 标准化日期格式
                    date_str = date_str.replace('年', '-').replace('月', '-').replace('/', '-')
                    return date_str
            
            # 如果找不到日期，返回空字符串
            return ""
        except:
            return ""
    
    def write_to_csv(self, data):
        """
        写入CSV文件（追加模式，与现有格式兼容）
        """
        if data is None:
            return
        
        # 检查文件是否存在，如果不存在则写入表头
        file_exists = os.path.exists(self.data_file)
        
        with open(self.data_file, 'a', encoding='utf-8-sig') as f:
            if not file_exists:
                # 写入表头
                f.write('期号,开奖日期,红球1,红球2,红球3,红球4,红球5,红球6,蓝球\n')
            
            # 写入数据
            f.write(f"{data['期号']},{data['开奖日期']},{data['红球1']},{data['红球2']},{data['红球3']},{data['红球4']},{data['红球5']},{data['红球6']},{data['蓝球']}\n")
    
    def fetch_from_official_api(self):
        """
        从官方API获取数据（优先使用，更稳定）
        """
        print("尝试从官方API获取数据...")
        
        try:
            params = {
                'name': 'ssq',
                'issueCount': '',
                'pageNo': '1',
                'pageSize': '9999',
                'systemType': 'PC'
            }
            
            response = requests.get(self.official_api, params=params, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('result'):
                    results = data['result']
                    print(f"从官方API获取到 {len(results)} 条数据")
                    
                    data_list = []
                    for item in results:
                        try:
                            period = item.get('code', '')
                            red_str = item.get('red', '')
                            blue_str = item.get('blue', '')
                            
                            if not period or not red_str or not blue_str:
                                continue
                            
                            # 解析红球（逗号分隔，如 "01,03,05,18,29,32"）
                            red_nums = [x.strip() for x in red_str.split(',')]
                            if len(red_nums) >= 6:
                                red_balls = [int(x) for x in red_nums[:6]]
                            else:
                                continue
                            
                            # 解析蓝球
                            blue_nums = re.findall(r'\d+', blue_str)
                            if blue_nums:
                                blue_ball = int(blue_nums[0])
                            else:
                                continue
                            
                            # 验证数据
                            if len(set(red_balls)) != 6 or any(r < 1 or r > 33 for r in red_balls):
                                continue
                            if blue_ball < 1 or blue_ball > 16:
                                continue
                            
                            red_balls.sort()
                            
                            # 清理日期格式（移除星期）
                            date = item.get('date', '').replace('(日)', '').replace('(一)', '').replace('(二)', '').replace('(三)', '').replace('(四)', '').replace('(五)', '').replace('(六)', '')
                            
                            data_list.append({
                                '期号': period,
                                '开奖日期': date,
                                '红球1': str(red_balls[0]),
                                '红球2': str(red_balls[1]),
                                '红球3': str(red_balls[2]),
                                '红球4': str(red_balls[3]),
                                '红球5': str(red_balls[4]),
                                '红球6': str(red_balls[5]),
                                '蓝球': str(blue_ball)
                            })
                        except Exception as e:
                            continue
                    
                    if data_list:
                        # 写入数据
                        new_count = 0
                        existing_periods = set()
                        
                        # 读取已有期号
                        if os.path.exists(self.data_file):
                            try:
                                with open(self.data_file, 'r', encoding='utf-8-sig') as f:
                                    lines = f.readlines()
                                    for line in lines[1:]:
                                        parts = line.strip().split(',')
                                        if parts:
                                            existing_periods.add(parts[0])
                            except:
                                pass
                        
                        # 写入新数据
                        for data in data_list:
                            if data['期号'] not in existing_periods:
                                self.write_to_csv(data)
                                new_count += 1
                        
                        print(f"✓ 成功从官方API获取并写入 {new_count} 条新数据")
                        return True
                        
        except Exception as e:
            print(f"从官方API获取数据失败: {e}")
        
        return False
    
    def turn_page(self):
        """
        获取所有期号列表并下载数据
        """
        print("开始从500.com获取双色球历史数据...")
        print(f"主页面: {self.base_url}")
        
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=15)
            response.encoding = 'gb2312'
            
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            pageList = soup.select("div.iSelectList a")
            
            total = len(pageList)
            print(f"找到 {total} 个期号链接")
            
            if total == 0:
                print("未找到期号链接，请检查网站结构是否改变")
                return
            
            # 如果文件已存在，询问是否删除
            if os.path.exists(self.data_file):
                print(f"\n文件 {self.data_file} 已存在")
                print("注意: 新数据将追加到现有文件，不会删除旧数据")
                print("如需重新获取，请先删除现有文件")
            
            success_count = 0
            fail_count = 0
            failed_items = []  # 记录失败的项，最后重试
            
            for i, p in enumerate(pageList, 1):
                url = p.get('href', '')
                page_text = p.string if p.string else p.get_text().strip()
                
                if not url or not page_text:
                    continue
                
                print(f"[{i}/{total}] 正在获取第{page_text}期...", end=' ')
                
                data = self.download(url, page_text)
                
                if data:
                    self.write_to_csv(data)
                    print(f"✓ 完成")
                    success_count += 1
                else:
                    print(f"✗ 失败")
                    fail_count += 1
                    failed_items.append((url, page_text, i))  # 记录失败项
                
                # 避免请求过快，增加延迟时间
                # 每10个请求后等待更长时间，让服务器有时间处理
                if i % 10 == 0:
                    time.sleep(1.0)  # 增加到1秒
                elif i % 5 == 0:
                    time.sleep(0.5)  # 每5个请求等待0.5秒
                else:
                    time.sleep(0.3)  # 增加到0.3秒
            
            # 重试失败的请求
            if failed_items:
                print(f"\n开始重试失败的 {len(failed_items)} 个请求...")
                retry_success = 0
                retry_fail = 0
                
                for url, page_text, original_index in failed_items:
                    print(f"  重试第{page_text}期...", end=' ')
                    time.sleep(2.0)  # 重试时等待更长时间
                    
                    data = self.download(url, page_text)
                    if data:
                        self.write_to_csv(data)
                        print(f"✓ 成功")
                        retry_success += 1
                        success_count += 1
                        fail_count -= 1
                    else:
                        print(f"✗ 仍然失败")
                        retry_fail += 1
                
                print(f"重试结果: 成功 {retry_success} 个，失败 {retry_fail} 个")
            
            print(f"\n完成！")
            print(f"成功: {success_count} 期")
            print(f"失败: {fail_count} 期")
            print(f"数据已保存到: {self.data_file}")
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def fetch_all(self):
        """
        获取所有历史数据的主方法
        优先使用官方API，失败则使用500.com
        """
        # 优先使用官方API
        if self.fetch_from_official_api():
            print("\n✓ 使用官方API成功获取数据")
            return
        
        # 如果官方API失败，使用500.com作为备用
        print("\n官方API不可用，使用500.com作为备用数据源...")
        self.turn_page()


def main():
    fetcher = SSQDataFetcher500Com()
    fetcher.fetch_all()


if __name__ == '__main__':
    main()

