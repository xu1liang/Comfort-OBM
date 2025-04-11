#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from test import (
    DrivingClassifier,
    DrivingDataset,
    prepare_time_series_data,
)  # 从训练脚本导入

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def load_model(model_path, input_dim=9):
    """加载与训练脚本完全兼容的模型"""
    # 注意：这里不再重新定义模型，直接使用训练脚本的类
    model = DrivingClassifier(
        input_dim=input_dim, d_model=input_dim * 20, nhead=input_dim, num_layers=2
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)

    # 调试信息
    print("\n模型参数检查:")
    print(f"实际加载的参数维度: { {k: v.shape for k, v in state_dict.items()} }")
    print(f"当前模型期望维度: { {k: v.shape for k, v in model.state_dict().items()} }")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def prepare_test_data():
    """准备测试数据（与训练脚本相同方式）"""
    train_scenarios = ["高速非变道晃动Good", "高速非变道晃动Hard"]

    test_scenarios = ["高速非变道晃动Good", "高速非变道晃动Hard"]

    # 使用训练脚本的函数准备数据
    train_sequences, train_labels, test_sequences, test_labels, input_dim, name_list = (
        prepare_time_series_data(train_scenarios, test_scenarios)
    )

    return DrivingDataset(test_sequences, test_labels), name_list


def predict_and_visualize(model, sequence, sample_name):
    """预测可视化函数（增加特征绘图）"""
    # 特征名称映射
    feature_names = [
        "v",
        "lat_acc",
        "lat_jerk",
        "kappa",
        "dkappa",
        "steer_angle",
        "steer_angle_rate",
        "w",
        "dw",
    ]

    # 准备输入数据
    inputs = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    masks = torch.ones(1, sequence.shape[0], dtype=torch.float32).to(device)

    # 执行预测
    with torch.no_grad():
        outputs = model(inputs, masks)
        prob = torch.softmax(outputs, dim=1)[0]
        prediction = "Hard" if outputs.argmax(dim=1).item() else "Good"

    # 创建画布
    plt.figure(figsize=(15, 8))

    # 1. 绘制转向角曲线
    plt.subplot(3, 1, 1)
    steer_angle = sequence[:, 5]  # 第5个特征是ego_steer_angle
    time_steps = np.arange(len(steer_angle)) * 0.1  # 假设每0.1秒一个数据点

    features_to_plot = [
        2,
    ]
    colors = ["b"]

    for idx, color in zip(features_to_plot, colors):
        plt.plot(
            time_steps,
            sequence[:, idx],
            color=color,
            label=feature_names[idx],
            alpha=0.7,
        )

    plt.title("Key Features")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)

    # 2. 绘制其他关键特征
    plt.subplot(3, 1, 2)
    features_to_plot = [
        5,
    ]
    colors = [
        "g",
    ]

    for idx, color in zip(features_to_plot, colors):
        plt.plot(
            time_steps,
            sequence[:, idx],
            color=color,
            label=feature_names[idx],
            alpha=0.7,
        )

    plt.title("Key Features")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)

    # 2. 绘制其他关键特征
    plt.subplot(3, 1, 3)
    features_to_plot = [
        6,
    ]
    colors = [
        "r",
    ]

    for idx, color in zip(features_to_plot, colors):
        plt.plot(
            time_steps,
            sequence[:, idx],
            color=color,
            label=feature_names[idx],
            alpha=0.7,
        )

    plt.title("Key Features")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)

    return {
        "sample_name": sample_name,
        "prediction": prediction,
        "confidence": prob.cpu().numpy(),
        "length": sequence.shape[0],
        "steer_stats": {  # 返回转向角统计
            "max": float(steer_angle.max()),
            "min": float(steer_angle.min()),
            "mean": float(steer_angle.mean()),
            "std": float(steer_angle.std()),
        },
    }


def main(model_path):
    """主函数"""
    model = load_model(model_path)
    test_dataset, name_list = prepare_test_data()

    while True:
        print("\n" + "=" * 50)
        print("1. 随机测试样本")
        print("2. 退出")
        choice = input("请选择: ").strip()

        if choice == "1":
            idx = random.randint(0, len(test_dataset) - 1)
            sequence, label, _ = test_dataset[idx]
            sample_name = name_list[idx]

            result = predict_and_visualize(model, sequence.numpy(), sample_name)

            print(f"\n样本: {result['sample_name']}")
            print(f"真实标签: {'Hard' if label else 'Good'}")
            print(f"预测结果: {result['prediction']}")
            print(
                f"置信度: Good={result['confidence'][0]:.2%}, Hard={result['confidence'][1]:.2%}"
            )

        elif choice == "2":
            break

        else:
            print("无效输入")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="训练好的模型路径 (.pth)")
    args = parser.parse_args()

    try:
        main(args.model)
    except Exception as e:
        print(f"错误: {str(e)}")
        print("请检查:")
        print("1. 模型路径是否正确")
        print("2. 训练脚本(test.py)中的模型定义是否与训练时一致")
        print("3. 数据准备函数输出格式是否匹配")
