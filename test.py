#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report

import time

from train.data_processing import (
    create_dataloaders,
    create_dataloaders_auto,
    prepare_test_data,
    prepare_time_series_data,
)
from train.model import DrivingClassifier, evaluate, train_model
from viz.viz import (
    plot_training_history,
    plot_auc_curves,
    visualize_positional_encoding,
    visualize_embeddings,
    plot_confusion_matrix,
)


def data_preparation():
    set_test_set = False  # 是否指定测试集
    if set_test_set:
        train_scenarios = [
            "高速非变道晃动Good",
        ]
        test_scenarios = ["高速非变道晃动Hard"]
        train_sequences, train_labels, test_sequences, test_labels, input_dim, _ = (
            prepare_time_series_data(train_scenarios, test_scenarios)
        )
        train_loader, val_loader, test_loader = create_dataloaders(
            train_sequences, train_labels, test_sequences, test_labels
        )
    else:
        X, y, input_dim = prepare_test_data()
        train_loader, val_loader, test_loader = create_dataloaders_auto(
            X, y, batch_size=32
        )
    return train_loader, val_loader, test_loader, input_dim


# ==================== 主程序 ====================
def main():
    # 1. 准备数据
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader, input_dim = data_preparation()
    # 5. 初始化模型
    print("\nInitializing model...")
    model = DrivingClassifier(
        input_dim=input_dim, d_model=input_dim * 20, nhead=input_dim, num_layers=2
    ).to(device)

    # 6. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )

    # 7. 训练模型
    print("\nStart training...")
    start_time = time.time()
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=30
    )
    print(f"\nTraining completed in {(time.time() - start_time) / 60:.1f} minutes")

    # 8. 可视化训练过程
    plot_training_history(history)

    # 9. 在测试集上评估
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # 10. 可视化测试结果
    plot_auc_curves(model, test_loader, "test", "Test Set")
    visualize_positional_encoding(input_dim * 20)
    visualize_embeddings(model, test_loader)

    # 生成预测结果用于混淆矩阵
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            masks = batch["masks"].to(device)
            outputs = model(inputs, masks)
            y_true.extend(batch["labels"].numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    plot_confusion_matrix(np.array(y_true), np.array(y_pred))

    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Shaking"]))


if __name__ == "__main__":
    main()
