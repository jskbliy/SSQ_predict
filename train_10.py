"""
训练使用10期数据的LSTM模型
使用最近10期数据预测下一期
"""
from data_processor_10 import SSQDataProcessor10
from lstm_model_10 import SSQLSTMModel10


def main():
    print("=" * 60)
    print("双色球LSTM模型训练（使用10期数据）")
    print("=" * 60)
    
    # 步骤1: 数据预处理
    print("\n步骤1: 数据预处理（使用10期数据）...")
    print("-" * 60)
    processor = SSQDataProcessor10()
    processor.process_data(seq_length=10, train_ratio=0.8)
    
    # 步骤2: 训练模型
    print("\n步骤2: 训练模型...")
    print("-" * 60)
    model = SSQLSTMModel10()
    model.train(epochs=200, batch_size=32)
    
    # 步骤3: 预测下一期
    print("\n步骤3: 预测下一期号码...")
    print("-" * 60)
    prediction = model.predict_next()
    
    print("\n" + "=" * 60)
    print("预测结果:")
    print("=" * 60)
    print(f"红球: {prediction['红球']}")
    print(f"蓝球: {prediction['蓝球']}")
    print("=" * 60)
    print("\n提示: 此预测仅供参考，彩票开奖具有随机性，请理性投注！")


if __name__ == "__main__":
    main()

