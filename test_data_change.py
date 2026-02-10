"""
测试数据更新后特征是否变化
用于验证数据读取和特征计算是否正确
"""
import pandas as pd
import numpy as np
from data_processor_10 import SSQDataProcessor10
import pickle
import hashlib


def main():
    print("=" * 60)
    print("测试：数据更新后特征是否变化")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_csv("ssq_history.csv", encoding='utf-8-sig')
    df = df.drop_duplicates(subset=['期号'], keep='last')
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)
    
    print(f"\n数据总期数: {len(df)}")
    print(f"最新期号: {df.iloc[-1]['期号']}")
    
    # 显示最近10期数据
    print("\n最近10期原始数据:")
    for i in range(len(df) - 10, len(df)):
        red = [df.iloc[i][f'红球{j+1}'] for j in range(6)]
        blue = df.iloc[i]['蓝球']
        period = df.iloc[i]['期号']
        print(f"  {period}: 红球{red}, 蓝球{blue}")
    
    # 计算特征
    processor = SSQDataProcessor10()
    features = processor.prepare_features(df)
    
    # 加载scaler
    with open("scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    features_scaled = scaler.transform(features)
    
    # 只取最近10期
    features_recent = features_scaled[-10:]
    
    print(f"\n特征形状: {features_recent.shape}")
    
    # 显示每期的特征值
    print("\n最近10期的特征（前7维，对应红球+蓝球）:")
    for i in range(10):
        period_idx = len(df) - 10 + i
        period = df.iloc[period_idx]['期号']
        print(f"  期号 {period}: {features_recent[i, :7]}")
    
    # 计算哈希值
    feature_hash = hashlib.md5(features_recent.tobytes()).hexdigest()[:8]
    print(f"\n特征数据哈希值: {feature_hash}")
    
    print("\n" + "=" * 60)
    print("测试说明:")
    print("=" * 60)
    print("1. 记录当前的哈希值")
    print("2. 更新 ssq_history.csv（添加新一期数据）")
    print("3. 再次运行此脚本")
    print("4. 对比哈希值是否变化")
    print("5. 如果哈希值变化 -> 特征确实在更新")
    print("6. 如果预测结果还是一样 -> 模型权重问题")
    print("=" * 60)
    
    print("\n解释:")
    print("-" * 60)
    print("• 特征变化但预测不变 = 模型权重固定导致")
    print("• 解决方法: 重新训练模型（python train_10.py）")
    print("-" * 60)


if __name__ == "__main__":
    main()


