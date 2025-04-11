import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
# 设备配置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 模型定义 ====================
class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
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
        return x + self.pos_embedding[: x.size(1)]  # 自动广播


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