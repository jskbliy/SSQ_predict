"""
使用已训练的10期模型进行预测
只预测，不训练
"""
from lstm_model_10 import SSQLSTMModel10
import os


def main():
    print("=" * 60)
    print("双色球LSTM模型预测（使用10期数据）")
    print("=" * 60)
    
    # 检查模型文件是否存在
    model_file = "ssq_lstm_model_10.weights.h5"
    if not os.path.exists(model_file):
        print(f"\n错误: 模型文件 {model_file} 不存在！")
        print("请先运行 train_10.py 训练模型。")
        return
    
    # 检查数据文件是否存在
    data_file = "processed_data_10.npz"
    if not os.path.exists(data_file):
        print(f"\n错误: 数据文件 {data_file} 不存在！")
        print("请先运行 train_10.py 处理数据并训练模型。")
        return
    
    print("\n重要说明:")
    print("-" * 60)
    print("• 此脚本使用已训练的模型进行预测")
    print("• 即使CSV数据更新，如果模型未重新训练，预测结果也不会变化")
    print("• 要获得基于最新数据的预测，请运行 train_10.py 重新训练模型")
    print("-" * 60)
    
    # 加载模型并预测
    print("\n正在加载模型...")
    model = SSQLSTMModel10()
    
    print("\n正在预测下一期号码...")
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

