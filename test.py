#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns
import time
import math
from typing import List, Dict, Tuple, Optional
from sim.api import ExtractFeatureMetrics

# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== 数据准备 ====================
class DrivingDataset(Dataset):
    """驾驶行为数据集类"""

    def __init__(self, sequences: List[np.ndarray], labels: np.ndarray):
        """
        Args:
            sequences: List of (seq_len, 9) arrays
            labels: (n_samples,) array of labels (0: Good, 1: Hard)
        """
        self.sequences = [torch.FloatTensor(x) for x in sequences]
        self.labels = torch.LongTensor(labels)
        self.lengths = [x.shape[0] for x in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]


def prepare_time_series_data(
    train_scenarios: List[str], test_scenarios: List[str]
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    """准备时间序列数据，按场景划分训练集和测试集"""
    required_features = [
        "ego_v",
        "ego_lat_acc",
        "ego_lat_jerk",
        "ego_kappa",
        "ego_dkappa",
        "ego_steer_angle",
        "ego_steer_angle_rate",
        "ego_w",
        "ego_dw",
    ]
    task_map = {
        "高速变道晃动Hard": "67f5e29f02daf0bcf4461bc1",
        "高速变道晃动Good": "67f5e29fd7c5690104005ee5",
        "高速非变道晃动Hard": "67f5e29ed7c5690104005eb7",
        "高速非变道晃动Good": "67f5e29ed7c5690104005eb6",
        "城区变道晃动Hard": "67f5e29e02daf0bcf4461bbb",
        "城区变道晃动Good": "67f5e29ed7c5690104005ed1",
        "城区非变道晃动Hard": "67f5e29ea236be4bdef709d3",
        "城区非变道晃动Good": "67f5e2a0a236be4bdef70a1c",
        "猛打场景库Hard": "67f5e29f02daf0bcf4461bd1",
        "猛打场景库Good": "67f5e29f02daf0bcf4461bf7",
    }

    def load_scenarios(scenario_names: List[str]):
        sequences = []
        labels = []
        sample_names = []  # 新增：用于存储样本名称

        for scenario_name in scenario_names:
            task_id = task_map[scenario_name]
            task_type = "Good" if "Good" in scenario_name else "Hard"

            try:
                feature_list, name_list = ExtractFeatureMetrics(
                    [task_id], "dd_sudden_turn"
                )
                print(f"从场景 {scenario_name} 加载到 {len(feature_list[0])} 条数据")

                # 确保feature_list和name_list一一对应
                assert len(feature_list[0]) == len(name_list[0]), (
                    "特征列表和名称列表长度不匹配"
                )

                for bag_data, bag_name in zip(feature_list[0], name_list[0]):
                    if not all(k in bag_data for k in required_features):
                        print(f"警告: 样本 {bag_name} 缺少必要特征")
                        continue

                    seq_length = len(bag_data[required_features[0]])
                    time_series = np.zeros((seq_length, len(required_features)))

                    for i, feat in enumerate(required_features):
                        assert len(bag_data[feat]) == seq_length, (
                            f"特征 {feat} 长度不一致"
                        )
                        time_series[:, i] = bag_data[feat]

                    sequences.append(time_series)
                    labels.append(0 if task_type == "Good" else 1)
                    sample_names.append(bag_name)  # 记录对应名称

            except Exception as e:
                print(f"Error processing {task_id}: {str(e)}")
                continue

        # 验证数据一致性
        assert len(sequences) == len(sample_names), "序列和名称数量不匹配"
        print(f"成功加载 {len(sequences)} 条有效数据")

        return sequences, np.array(labels), sample_names

    # 加载训练集和测试集
    train_sequences, train_labels, name_list = load_scenarios(train_scenarios)
    test_sequences, test_labels, name_list = load_scenarios(test_scenarios)

    # 数据统计
    def print_stats(name, sequences, labels):
        seq_lengths = [seq.shape[0] for seq in sequences]
        print(f"\n{'=' * 40}")
        print(f"{name} Statistics")
        print(f"{'=' * 40}")
        print(f"Total sequences: {len(sequences)}")
        print(f"Good driving: {sum(labels == 0)}")
        print(f"Hard driving: {sum(labels == 1)}")
        print(f"Avg sequence length: {np.mean(seq_lengths):.1f} steps")
        print(
            f"Duration range: {min(seq_lengths) * 0.1:.1f}s - {max(seq_lengths) * 0.1:.1f}s"
        )
        print(f"Scenarios: {[s.split('_')[0] for s in set(train_scenarios)]}")
        print(f"{'=' * 40}")

    print_stats("Training Set", train_sequences, train_labels)
    print_stats("Test Set", test_sequences, test_labels)

    return (
        train_sequences,
        train_labels,
        test_sequences,
        test_labels,
        len(required_features),
        name_list,
    )

def prepare_test_data():
    """准备时间序列数据"""
    task_map = {
        "高速变道晃动Hard": "67f5e29f02daf0bcf4461bc1",
        "高速变道晃动Good": "67f5e29fd7c5690104005ee5",
        "高速非变道晃动Hard": "67f5e29ed7c5690104005eb7",
        "高速非变道晃动Good": "67f5e29ed7c5690104005eb6",
        "城区变道晃动Hard": "67f5e29e02daf0bcf4461bbb",
        "城区变道晃动Good": "67f5e29ed7c5690104005ed1",
        "城区非变道晃动Hard": "67f5e29ea236be4bdef709d3",
        "城区非变道晃动Good": "67f5e2a0a236be4bdef70a1c",
        # "猛打场景库Hard": "67f5e29f02daf0bcf4461bd1",
        # "猛打场景库Good": "67f5e29f02daf0bcf4461bf7",
    }

    good_sequences = []
    bad_sequences = []
    required_features = [
        "ego_v",
        "ego_lat_acc",
        "ego_lat_jerk",
        "ego_kappa",
        "ego_dkappa",
        "ego_steer_angle",
        "ego_steer_angle_rate",
        "ego_w",
        "ego_dw",
    ]

    for task_type in ["Good", "Hard"]:
        for scenario_name, task_id in task_map.items():
            if task_type not in scenario_name:
                continue

            try:
                feature_list, _ = ExtractFeatureMetrics([task_id], "dd_sudden_turn")

                for bag_data in feature_list[0]:
                    # 检查特征完整性
                    if not all(k in bag_data for k in required_features):
                        continue

                    # 转换时间序列数据 (T, 9)
                    seq_length = len(bag_data[required_features[0]])
                    time_series = np.zeros((seq_length, len(required_features)))

                    for i, feat in enumerate(required_features):
                        # 确保每个特征长度一致
                        assert len(bag_data[feat]) == seq_length
                        time_series[:, i] = bag_data[feat]

                    # 根据任务类型存储
                    if task_type == "Good":
                        good_sequences.append(time_series)
                    else:
                        bad_sequences.append(time_series)

            except Exception as e:
                print(f"Error processing {task_id}: {str(e)}")
                continue

    # 创建标签
    X = good_sequences + bad_sequences
    y = np.array([0] * len(good_sequences) + [1] * len(bad_sequences))
    
    return X, y, len(required_features)

def create_dataloaders_auto(
    X: List[np.ndarray],
    y: np.ndarray,
    batch_size: int = 32,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    dataset = DrivingDataset(X, y)

    # 划分数据集
    val_size = int(val_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(
        f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}"
    )

    # 定义collate_fn处理不等长序列
    def collate_fn(batch):
        inputs = [x[0] for x in batch]
        labels = torch.stack([x[1] for x in batch])
        lengths = torch.tensor([x[2] for x in batch])

        # 填充序列 (batch, max_len, 9)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

        # 创建mask (1表示有效数据，0表示padding)
        max_len = padded_inputs.shape[1]
        masks = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(
            1
        )

        return {"inputs": padded_inputs, "labels": labels, "masks": masks.float()}

    # 创建数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True
    )

    print(
        f"Created dataloaders - Train: {len(train_loader)} batches, "
        f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader


def create_dataloaders(
    train_sequences: List[np.ndarray],
    train_labels: np.ndarray,
    test_sequences: List[np.ndarray],
    test_labels: np.ndarray,
    batch_size: int = 32,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器，从训练集中划分验证集"""
    train_dataset = DrivingDataset(train_sequences, train_labels)

    # 划分训练集和验证集
    val_size = int(val_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_set, val_set = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_dataset = DrivingDataset(test_sequences, test_labels)

    print(
        f"\nDataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_dataset)}"
    )

    def collate_fn(batch):
        inputs = [x[0] for x in batch]
        labels = torch.stack([x[1] for x in batch])
        lengths = torch.tensor([x[2] for x in batch])

        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        max_len = padded_inputs.shape[1]
        masks = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(
            1
        )

        return {
            "inputs": padded_inputs,
            "labels": labels,
            "masks": masks.float(),
            "speeds": padded_inputs[:, :, 0].mean(dim=1),  # 添加速度信息
        }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ==================== 模型定义 ====================
class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))  # 可学习参数
        self.dropout = nn.Dropout(p=0.1)  # 可选

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:x.size(1)]  # 自动广播


class DrivingClassifier(nn.Module):
    """驾驶行为分类模型"""

    def __init__(
        self,
        input_dim: int = 9,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
    ):
        super().__init__()

        # 输入处理
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.BatchNorm1d(d_model), nn.ReLU()
        )

        # 位置编码
        pos_encoding_type = "learned"  # 可选: "original" 或 "learned"
        if pos_encoding_type == "original":
            self.pos_encoder = PositionalEncoding(d_model)
        elif pos_encoding_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch, seq_len, input_dim=9)

        # 1. 投影层处理
        # 先reshape为(batch*seq_len, input_dim)以适应线性层
        batch_size, seq_len, input_dim = x.shape
        x = x.reshape(-1, input_dim)  # (batch*seq_len, 9)
        x = self.input_proj(x)  # (batch*seq_len, d_model)

        # 恢复序列结构 (batch, seq_len, d_model)
        x = x.reshape(batch_size, seq_len, -1)

        # 2. 位置编码
        x = self.pos_encoder(x)

        # 3. Transformer处理
        x = self.transformer(x, src_key_padding_mask=~mask.bool())

        # 4. 取序列最后有效时间步
        last_idx = mask.sum(dim=1).long() - 1
        last_output = x[torch.arange(batch_size), last_idx]  # (batch, d_model)

        # 5. 分类头
        return self.classifier(last_output)


# ==================== 训练与评估 ====================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epochs: int = 20,
    model_save_path: str = "best_model.pth",
) -> Dict[str, List[float]]:
    """模型训练"""
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["masks"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 记录历史
        history["train_loss"].append(train_loss / len(train_loader.dataset))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 学习率调整
        if scheduler:
            scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train Loss: {history['train_loss'][-1]:.4f}, "
            f"Val Loss: {history['val_loss'][-1]:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

    return history


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: Optional[nn.Module] = None
) -> Tuple[float, float]:
    """模型评估"""
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["labels"].to(device)
            masks = batch["masks"].to(device)

            outputs = model(inputs, masks)
            if criterion:
                total_loss += criterion(outputs, labels).item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader.dataset) if criterion else 0.0
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


# ==================== 可视化工具 ====================
def visualize_positional_encoding(d_model=64, max_len=500):
    """独立可视化位置编码"""
    pe = PositionalEncoding(d_model, max_len)
    plt.figure(figsize=(15, 5))
    
    # 绘制正弦波示例
    plt.subplot(1, 2, 1)
    for i in range(0, d_model, d_model//4):  # 每25%显示一个维度
        plt.plot(pe.pe[:300, i].numpy(), 
                label=f'dim {i}')
    plt.title('Positional Encoding (Sine Waves)')
    plt.xlabel('Time Step')
    plt.ylabel('Encoding Value')
    plt.legend()
    plt.grid(True)
    
    # 绘制2D热图
    plt.subplot(1, 2, 2)
    plt.imshow(pe.pe[:100, :].numpy().T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('2D Encoding Matrix')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Dimension')
    
    plt.tight_layout()
    plt.savefig('./fig/pos_encoding_dual.png', dpi=300)
    plt.show(block=False)
    print("Positional encoding visualization completed")
    
def visualize_embeddings(
    model: nn.Module, loader: DataLoader, save_path: str = "./fig/embedding_visualization.png"
):
    """可视化特征嵌入空间"""
    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            masks = batch["masks"].to(device)

            # 使用与训练相同的前向传播方式
            batch_size, seq_len, input_dim = inputs.shape

            # 1. 投影层处理
            x = inputs.reshape(-1, input_dim)  # (batch*seq_len, 9)
            x = model.input_proj(x)  # (batch*seq_len, d_model)
            x = x.reshape(batch_size, seq_len, -1)  # (batch, seq_len, d_model)

            # 2. 位置编码
            x = model.pos_encoder(x)

            # 3. Transformer处理
            x = model.transformer(x, src_key_padding_mask=~masks.bool())

            # 取全局平均特征
            feat = x.mean(dim=1)  # (batch, d_model)

            embeddings.append(feat.cpu())
            labels.append(batch["labels"].cpu())

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=20, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    # 绘制
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="viridis", alpha=0.6
    )
    plt.colorbar(scatter, ticks=[0, 1], label="Class (0: Normal, 1: Shaking)")
    plt.title("t-SNE Visualization of Driving Behavior Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Embedding visualization saved to {save_path}")


def plot_training_history(
    history: Dict[str, List[float]], save_path: str = "./fig/training_history.png"
):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "./fig/confusion_matrix.png"
):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Shaking"],
        yticklabels=["Normal", "Shaking"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_feature_distribution(
    X: List[np.ndarray],
    y: np.ndarray,
    feature_idx: int = 0,
    feature_name: str = "ego_v",
    save_path: str = "./fig/feature_dist.png",
    n_examples: int = 10,  # 新增参数：展示的样例序列数量
    example_length: int = 50,  # 展示的序列片段长度
    seed: int = 42,  # 随机种子
):
    """
    绘制特征分布图（增强版）

    参数:
        X: 输入序列列表 [n_samples, seq_len, n_features]
        y: 标签数组 (0: Good, 1: Hard)
        feature_idx: 要可视化的特征索引
        feature_name: 特征名称（用于标题）
        save_path: 图片保存路径
        n_examples: 每类展示的样例序列数量
        example_length: 展示的序列长度（时间步）
        seed: 随机种子
    """
    np.random.seed(seed)
    plt.figure(figsize=(12, 6 + n_examples))  # 动态调整高度

    try:
        # ==================== 第一部分：特征值分布 ====================
        plt.subplot(2, 1, 1)

        # 准备数据
        good_values = []
        hard_values = []
        for seq, label in zip(X, y):
            if label == 0:
                good_values.extend(seq[:, feature_idx].tolist())
            else:
                hard_values.extend(seq[:, feature_idx].tolist())

        # 绘制增强版小提琴图
        violin = plt.violinplot(
            [good_values, hard_values],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
        )

        # 设置颜色
        colors = ["blue", "red"]
        for pc, color in zip(violin["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for part in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
            if part in violin:
                violin[part].set_color("black")

        # 添加统计信息
        for i, (values, color) in enumerate(zip([good_values, hard_values], colors)):
            mean = np.mean(values)
            std = np.std(values)
            plt.text(
                i + 1,
                np.max(values) * 1.05,
                f"μ={mean:.2f} σ={std:.2f}",
                ha="center",
                color=color,
            )

        plt.xticks([1, 2], ["Good", "Hard"])
        plt.title(f"{feature_name} Distribution (All Samples)")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)

        # ==================== 第二部分：多序列可视化 ====================
        plt.subplot(2, 1, 2)

        # 获取随机样例
        good_indices = np.where(y == 0)[0]
        hard_indices = np.where(y == 1)[0]

        # 确保不超出范围
        n_good = min(n_examples, len(good_indices))
        n_hard = min(n_examples, len(hard_indices))

        selected_good = np.random.choice(good_indices, n_good, replace=False)
        selected_hard = np.random.choice(hard_indices, n_hard, replace=False)

        # 绘制Good序列
        for i, idx in enumerate(selected_good):
            seq = X[idx][:, feature_idx]
            # 随机选取片段
            start = np.random.randint(0, max(1, len(seq) - example_length))
            segment = seq[start : start + example_length]
            plt.plot(segment, "b-", alpha=0.7, label="Good" if i == 0 else "")

        # 绘制Hard序列
        for i, idx in enumerate(selected_hard):
            seq = X[idx][:, feature_idx]
            start = np.random.randint(0, max(1, len(seq) - example_length))
            segment = seq[start : start + example_length]
            plt.plot(segment, "r-", alpha=0.7, label="Hard" if i == 0 else "")

        # 美化图形
        plt.title(f"Random {feature_name} Sequences (n={n_examples} per class)")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Enhanced feature distribution saved to {save_path}")

    except Exception as e:
        print(f"Error plotting feature distribution: {str(e)}")
        plt.close()

def visualize_feature_distribution(
    train_sequences: List[np.ndarray],
    train_labels: np.ndarray,
):
    # 3. 可视化原始数据特征
    print("\nVisualizing feature distributions...")
    feature_info = [
        (
            0,
            "ego_v (Speed)",
        ),
        (
            1,
            "ego_lat_acc (Lateral Acceleration)",
        ),
        (
            2,
            "ego_lat_jerk (Lateral Jerk)",
        ),
        (
            3,
            "ego_kappa (Curvature)",
        ),
        (
            4,
            "ego_dkappa (Curvature Rate)",
        ),
        (
            5,
            "ego_steer_angle (Steering Angle)",
        ),
        (
            6,
            "ego_steer_angle_rate (Steering Rate)",
        ),
        (
            7,
            "ego_w (Yaw Rate)",
        ),
        (
            8,
            "ego_dw (Yaw Acceleration)",
        ),
    ]
    for idx, name in feature_info:
        plot_feature_distribution(
            train_sequences,
            train_labels,
            idx,
            name,
            f"train_{name.split()[0]}_dist.png",
        )


def plot_auc_curves(
    model: nn.Module,
    loader: DataLoader,
    save_prefix: str = "eval",
    title_suffix: str = "",
):
    """
    绘制ROC曲线和PR曲线，并计算AUC值
    参数:
        model: 训练好的模型
        loader: 数据加载器 (val或test)
        save_prefix: 图片保存前缀
        title_suffix: 标题后缀 (如"Validation"或"Test")
    """
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            masks = batch["masks"].to(device)

            outputs = model(inputs, masks)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 取Hard类的概率

            y_true.extend(batch["labels"].numpy())
            y_score.extend(probabilities.cpu().numpy())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # ========== ROC曲线 ==========
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {title_suffix}")
    plt.legend(loc="lower right")

    # ========== PR曲线 ==========
    plt.subplot(1, 2, 2)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    plt.plot(
        recall, precision, color="blue", lw=2, label=f"PR curve (AP = {pr_auc:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve {title_suffix}")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"./fig/{save_prefix}_auc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"AUC curves saved to {save_prefix}_auc_curves.png")

def data_preparation():
    set_test_set = False  # 是否指定测试集
    if set_test_set:
        train_scenarios = ["高速非变道晃动Good", "高速非变道晃动Hard"]
        test_scenarios = ["城区非变道晃动Good", "城区非变道晃动Hard"]
        train_sequences, train_labels, test_sequences, test_labels, input_dim, _ = (
            prepare_time_series_data(train_scenarios, test_scenarios)
        )
        train_loader, val_loader, test_loader = create_dataloaders(
            train_sequences, train_labels, test_sequences, test_labels
        )
    else:
        X, y, input_dim = prepare_test_data()
        train_loader, val_loader, test_loader = create_dataloaders_auto(X, y, batch_size=32)
    return train_loader, val_loader, test_loader, input_dim

# ==================== 主程序 ====================
def main():
    # 1. 准备数据
    
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
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20
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
