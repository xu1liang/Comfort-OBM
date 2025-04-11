import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Tuple

from sim.api import ExtractFeatureMetrics

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
    task_map = {
        "高速变道晃动Hard": "67f5e29f02daf0bcf4461bc1",
        "高速变道晃动Good": "67f5e29fd7c5690104005ee5",
        "高速非变道晃动Hard": "67f7d0dfc4361c30d5faf75f",
        "高速非变道晃动Good": "67f7d0dcea73163ad7436880",
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
        "高速非变道晃动Hard": "67f7d0dfc4361c30d5faf75f",
        "高速非变道晃动Good": "67f7d0dcea73163ad7436880",
    }

    good_sequences = []
    bad_sequences = []
    required_features = [
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