"""
双色球数据预处理脚本（使用10期数据）
使用最近10期数据预测下一期
例如：1-10期预测11期，2-11期预测12期，3-12期预测13期...
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class SSQDataProcessor10:
    def __init__(self, data_file="ssq_history.csv"):
        self.data_file = data_file
        self.scaler_file = "scaler.pkl"
        self.processed_data_file = "processed_data_10.npz"  # 使用processed_data_10.npz格式
        self.seq_length = 10  # 使用10期数据
        
    def load_data(self):
        """加载原始数据"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件 {self.data_file} 不存在，请先运行 data_fetcher.py")
        
        df = pd.read_csv(self.data_file, encoding='utf-8-sig')
        print(f"加载了 {len(df)} 期数据")
        return df
    
    def prepare_features(self, df):
        """
        准备增强的特征数据
        包括：基础特征 + 频率特征 + 趋势特征 + 相关性特征
        """
        red_cols = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
        blue_col = '蓝球'
        
        red_balls = df[red_cols].astype(int).values
        blue_balls = df[blue_col].astype(int).values
        
        # 计算历史频率（用于频率特征）
        number_frequency = {}  # 记录每个号码的出现频率
        for num in range(1, 34):
            number_frequency[num] = []
        
        features = []
        for i in range(len(df)):
            red = red_balls[i]
            blue = blue_balls[i]
            
            red_sorted = sorted(red)
            
            # ========== 基础统计特征 ==========
            red_sum = sum(red_sorted)
            red_mean = np.mean(red_sorted)
            red_std = np.std(red_sorted)
            red_max = max(red_sorted)
            red_min = min(red_sorted)
            red_range = red_max - red_min
            red_median = np.median(red_sorted)
            red_odd_count = sum(1 for x in red_sorted if x % 2 == 1)
            red_even_count = 6 - red_odd_count
            red_small_count = sum(1 for x in red_sorted if x <= 17)
            red_large_count = 6 - red_small_count
            sum_zone = red_sum // 50
            
            # ========== 频率特征（最近N期的出现频率）==========
            # 计算每个号码在最近10期的出现频率
            freq_window = min(10, i)  # 使用最近10期，如果数据不足则用全部
            if freq_window > 0:
                recent_reds = red_balls[max(0, i-freq_window):i].flatten()
                number_counts = {}
                for num in range(1, 34):
                    number_counts[num] = np.sum(recent_reds == num)
                # 当前期号码的频率特征
                current_freq = [number_counts[num] for num in red_sorted]
                avg_freq = np.mean(current_freq)
                max_freq = max(current_freq) if current_freq else 0
                min_freq = min(current_freq) if current_freq else 0
            else:
                current_freq = [0] * 6
                avg_freq = 0
                max_freq = 0
                min_freq = 0
            
            # ========== 趋势特征 ==========
            # 计算号码的移动平均和趋势
            if i >= 3:
                # 3期移动平均
                recent_reds = red_balls[i-3:i]
                ma3_sum = np.mean([np.sum(r) for r in recent_reds])
                ma3_mean = np.mean([np.mean(r) for r in recent_reds])
                trend_sum = red_sum - ma3_sum  # 和值趋势
                trend_mean = red_mean - ma3_mean  # 均值趋势
            else:
                ma3_sum = red_sum
                ma3_mean = red_mean
                trend_sum = 0
                trend_mean = 0
            
            if i >= 5:
                # 5期移动平均
                recent_reds = red_balls[i-5:i]
                ma5_sum = np.mean([np.sum(r) for r in recent_reds])
                ma5_mean = np.mean([np.mean(r) for r in recent_reds])
            else:
                ma5_sum = red_sum
                ma5_mean = red_mean
            
            # ========== 相关性特征 ==========
            # 号码之间的间隔
            intervals = [red_sorted[j+1] - red_sorted[j] for j in range(5)]
            avg_interval = np.mean(intervals)
            max_interval = max(intervals)
            min_interval = min(intervals)
            interval_std = np.std(intervals)
            
            # 连号数量（相邻号码差为1）
            consecutive_count = sum(1 for interval in intervals if interval == 1)
            
            # 号码分布特征（三个区间：1-11, 12-22, 23-33）
            zone1_count = sum(1 for x in red_sorted if 1 <= x <= 11)
            zone2_count = sum(1 for x in red_sorted if 12 <= x <= 22)
            zone3_count = sum(1 for x in red_sorted if 23 <= x <= 33)
            
            # 号码跨度特征（最大最小值的差）
            span_ratio = red_range / 33.0  # 跨度比例
            
            # ========== 组合特征 ==========
            # 红球与蓝球的和
            total_sum = red_sum + blue
            
            # 红球与蓝球的差值
            red_blue_diff = abs(red_mean - blue)
            
            # 号码的方差
            red_variance = np.var(red_sorted)
            
            # ========== 构建特征向量 ==========
            feature_vector = [
                # 基础特征（7个号码）
                *red_sorted,
                blue,
                # 基础统计特征（12个）
                red_sum,
                red_mean,
                red_std,
                red_median,
                red_range,
                red_odd_count,
                red_even_count,
                red_small_count,
                red_large_count,
                sum_zone,
                red_variance,
                # 频率特征（4个）
                avg_freq,
                max_freq,
                min_freq,
                np.std(current_freq) if current_freq else 0,
                # 趋势特征（6个）
                ma3_sum,
                ma3_mean,
                ma5_sum,
                ma5_mean,
                trend_sum,
                trend_mean,
                # 相关性特征（8个）
                avg_interval,
                max_interval,
                min_interval,
                interval_std,
                consecutive_count,
                zone1_count,
                zone2_count,
                zone3_count,
                span_ratio,
                # 组合特征（3个）
                total_sum,
                red_blue_diff,
            ]
            
            features.append(feature_vector)
            
            # 更新频率统计
            for num in red_sorted:
                number_frequency[num].append(i)
        
        return np.array(features)
    
    def create_sequences(self, data, seq_length=10):
        """
        创建时间序列数据（使用固定10期）
        seq_length: 使用前多少期数据来预测下一期
        """
        X, y = [], []
        
        for i in range(seq_length, len(data)):
            # 输入：前seq_length期的特征
            X.append(data[i-seq_length:i])
            # 输出：第i期的红球和蓝球（前7个是红球和蓝球，后面是统计特征）
            y.append(data[i][:7])  # 只预测号码，不预测统计特征
        
        return np.array(X), np.array(y)
    
    def process_data(self, seq_length=10, train_ratio=0.8):
        """
        处理数据并保存
        seq_length: 使用前多少期数据来预测下一期（默认10期）
        train_ratio: 训练集比例
        """
        print("开始处理数据（使用固定10期序列）...")
        
        # 加载数据
        df = self.load_data()
        
        # 按时间顺序排序（最老的在前）
        df = df.sort_values('期号', ascending=True)
        
        # 准备特征
        features = self.prepare_features(df)
        print(f"特征维度: {features.shape}")
        
        # 数据标准化
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        
        # 保存scaler
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler已保存到 {self.scaler_file}")
        
        # 创建序列
        X, y = self.create_sequences(features_scaled, seq_length)
        print(f"创建了 {len(X)} 个训练样本")
        print(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 划分训练集和测试集
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        # 保存处理后的数据
        np.savez(
            self.processed_data_file,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            features_scaled=features_scaled,
            seq_length=seq_length
        )
        print(f"处理后的数据已保存到 {self.processed_data_file}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'features_scaled': features_scaled,
            'seq_length': seq_length
        }


if __name__ == "__main__":
    processor = SSQDataProcessor10()
    data_dict = processor.process_data(seq_length=10, train_ratio=0.8)

