"""
训练使用50期数据的LSTM模型
"""
from data_processor_50 import SSQDataProcessor50
from lstm_model_50 import SSQLSTMModel50


def main():
    print("=" * 60)
    print("双色球LSTM模型训练（使用50期数据）")
    print("=" * 60)
    
    # 步骤1: 数据预处理
    print("\n步骤1: 数据预处理（使用50期数据）...")
    print("-" * 60)
    processor = SSQDataProcessor50()
    processor.process_data(min_seq_length=50, train_ratio=0.8, max_seq_length=50)
    
    # 步骤2: 训练模型
    print("\n步骤2: 训练模型...")
    print("-" * 60)
    model = SSQLSTMModel50()
    model.train(epochs=200, batch_size=32)
    
    # 步骤3: 预测下一期
    print("\n步骤3: 预测下一期号码...")
    print("-" * 60)
    predictions = model.predict_next()
    
    # 新的返回格式包含三种策略的结果
    print("\n" + "=" * 60)
    print("预测结果汇总:")
    print("=" * 60)
    
    # 显示三种策略的结果
    for strategy_name, result in predictions.items():
        print(f"\n{strategy_name}:")
        print(f"  红球: {result['红球']}")
        print(f"  蓝球: {result['蓝球']}")
        print(f"  说明: {result['说明']}")
    
    # 默认推荐使用策略2（贪心选择）
    recommended = predictions.get('策略2_贪心选择', list(predictions.values())[0])
    print("\n" + "=" * 60)
    print("推荐使用（策略2：贪心选择）:")
    print("=" * 60)
    print(f"红球: {recommended['红球']}")
    print(f"蓝球: {recommended['蓝球']}")
    print("=" * 60)
    print("\n提示: 此预测仅供参考，彩票开奖具有随机性，请理性投注！")


if __name__ == "__main__":
    main()

