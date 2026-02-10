"""
验证预测逻辑：确认是否用最近10期预测下一期
"""
import pandas as pd
import numpy as np
from data_processor_10 import SSQDataProcessor10
import pickle

def main():
    print("=" * 60)
    print("验证预测逻辑")
    print("=" * 60)
    
    # 读取数据
    df = pd.read_csv("ssq_history.csv", encoding='utf-8-sig')
    df = df.drop_duplicates(subset=['期号'], keep='last')
    df = df.sort_values('期号', ascending=True).reset_index(drop=True)
    
    print(f"\n数据总期数: {len(df)}")
    print(f"最新期号: {df.iloc[-1]['期号']}")
    
    # 显示最近10期
    print(f"\n最近10期数据:")
    for i in range(len(df) - 10, len(df)):
        period = df.iloc[i]['期号']
        red = [df.iloc[i][f'红球{j+1}'] for j in range(6)]
        blue = df.iloc[i]['蓝球']
        print(f"  第{i+1}行 (期号{period}): 红球{red}, 蓝球{blue}")
    
    # 计算特征
    processor = SSQDataProcessor10()
    features = processor.prepare_features(df)
    
    # 加载scaler
    with open("scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    features_scaled = scaler.transform(features)
    
    # 显示最近10期的特征
    print(f"\n最近10期的特征（用于预测）:")
    for i in range(len(df) - 10, len(df)):
        period = df.iloc[i]['期号']
        feature_idx = i
        print(f"  期号 {period} (特征索引{feature_idx}): 前7维={features_scaled[feature_idx, :7]}")
    
    print(f"\n" + "=" * 60)
    print("预测逻辑说明:")
    print("=" * 60)
    print("训练时:")
    print("  - 样本1: 用第1-10期 → 预测第11期")
    print("  - 样本2: 用第2-11期 → 预测第12期")
    print("  - 样本3: 用第3-12期 → 预测第13期")
    print("  - ...")
    print("\n预测时:")
    print(f"  - 用最近10期（第{len(df)-9}期到第{len(df)}期）→ 预测第{len(df)+1}期")
    print(f"  - 即：用期号 {df.iloc[-10]['期号']} 到 {df.iloc[-1]['期号']} → 预测下一期")
    print("=" * 60)

if __name__ == "__main__":
    main()

