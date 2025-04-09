#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sim.api import ExtractFeatureMetrics
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import psutil
from tqdm import tqdm

# 1. 数据准备 ==============================================================
class DrivingBehaviorDataset(Dataset):
    """处理变长时序数据的Dataset类"""
    def __init__(self, feature_list, max_seq_len=100):
        """
        Args:
            feature_list: 原始特征列表，每个元素是包含多个特征的字典
            max_seq_len: 统一序列长度
        """
        # 提取特征
        self.sequences = []
        self.labels = []
        
        # 第一组数据标记为良好（1），第二组为不良（0）
        for i, group in enumerate(feature_list):
            for data in group:
                seq = data["ego_steer_angle"]  # 示例特征，替换为您的实际特征
                self.sequences.append(seq)
                self.labels.append(i)
        
        # 标准化处理
        self.scaler = StandardScaler()
        padded_seqs = np.zeros((len(self.sequences), max_seq_len))
        for idx, seq in enumerate(self.sequences):
            seq = np.array(seq)
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            else:
                seq = np.pad(seq, (0, max_seq_len - len(seq)), 'constant')
            padded_seqs[idx] = seq
        
        self.features = self.scaler.fit_transform(padded_seqs)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]), 
            torch.LongTensor([self.labels[idx]])
        )

# 2. Transformer模型 ======================================================
class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DrivingBehaviorTransformer(nn.Module):
    """驾驶行为分类Transformer"""
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, num_layers, max_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 二分类输出
        )

    def forward(self, src):
        # src形状: (seq_len, batch_size, input_dim)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src)
        output = output.mean(dim=0)  # 全局平均池化
        return self.classifier(output)

# 3. 训练函数 =============================================================
def train_model(model, train_loader, test_loader, device, epochs=50):
    """模型训练与评估"""
    # Mac优化配置
    if torch.backends.mps.is_available():
        model = model.to('mps')
        print("Using MPS acceleration")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            x, y = x.to(device), y.to(device).squeeze()
            optimizer.zero_grad()
            output = model(x.unsqueeze(1))  # 添加特征维度
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 评估阶段
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).squeeze()
                output = model(x.unsqueeze(1))
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
        
        acc = 100 * correct / len(test_loader.dataset)
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_transformer.pth')
    
    return model

# 4. 主程序 ===============================================================
def main():
    # 配置参数
    config = {
        'max_seq_len': 100,      # 统一序列长度
        'input_dim': 1,          # 输入特征维度
        'd_model': 128,          # Transformer隐藏层维度
        'nhead': 4,              # 注意力头数
        'dim_feedforward': 256,   # FFN层维度
        'num_layers': 3,         # Transformer层数
        'batch_size': 32,        # 根据内存调整
        'epochs': 50
    }
    
    # 内存优化
    mem = psutil.virtual_memory()
    if mem.available < 8 * 1024**3:  # 小于8GB可用内存
        config['batch_size'] = 16
        print("Reduced batch_size due to memory constraints")

    # 加载数据 (替换为您的实际数据加载逻辑)
    goodBehaviorTaskId = "678528ae301d3d18a04b1570"
    badBehaviorTaskId = "678528c68499f11f18c77416"
    checker_name = "dd_sudden_turn"
    feature_list, _ = ExtractFeatureMetrics(
        [goodBehaviorTaskId, badBehaviorTaskId], checker_name
    )
    
    # 创建数据集
    full_dataset = DrivingBehaviorDataset(feature_list, config['max_seq_len'])
    train_data, test_data = train_test_split(
        list(range(len(full_dataset))), 
        test_size=0.2, 
        random_state=42,
        stratify=full_dataset.labels
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_data)
    test_dataset = torch.utils.data.Subset(full_dataset, test_data)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size']
    )

    # 初始化模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DrivingBehaviorTransformer(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        num_layers=config['num_layers'],
        max_len=config['max_seq_len']
    ).to(device)
    
    # 训练与评估
    train_model(model, train_loader, test_loader, device, config['epochs'])

if __name__ == "__main__":
    main()