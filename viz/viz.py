# ==================== 可视化工具 ====================
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import seaborn as sns
from train.model import PositionalEncoding

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def visualize_positional_encoding(d_model=64, max_len=500):
    """独立可视化位置编码"""
    pe = PositionalEncoding(d_model, max_len)
    plt.figure(figsize=(15, 5))

    # 绘制正弦波示例
    plt.subplot(1, 2, 1)
    for i in range(0, d_model, d_model // 4):  # 每25%显示一个维度
        plt.plot(pe.pe[:300, i].numpy(), label=f"dim {i}")
    plt.title("Positional Encoding (Sine Waves)")
    plt.xlabel("Time Step")
    plt.ylabel("Encoding Value")
    plt.legend()
    plt.grid(True)

    # 绘制2D热图
    plt.subplot(1, 2, 2)
    plt.imshow(pe.pe[:100, :].numpy().T, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title("2D Encoding Matrix")
    plt.xlabel("Time Step")
    plt.ylabel("Feature Dimension")

    plt.tight_layout()
    plt.savefig("./fig/pos_encoding_dual.png", dpi=300)
    plt.show(block=False)
    print("Positional encoding visualization completed")


def visualize_embeddings(
    model: nn.Module,
    loader: DataLoader,
    save_path: str = "./fig/embedding_visualization.png",
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
    print(f"Embeddings shape: {embeddings.shape}")  # 查看实际样本数量
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "./fig/confusion_matrix.png",
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
