import torch
from torch_geometric.data import HeteroData
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd


class StockHeteroDataProcessor:
    """
    股票异构图数据处理器
    """
    def __init__(self, window_size: int = 15, prediction_horizons: List[int] = [5, 30]):
        self.window_size = window_size
        self.prediction_horizons = prediction_horizons
    
    def create_hetero_data_from_features(self, 
                                       stock_features: torch.Tensor,  # [num_stocks, stock_feature_dim]
                                       comment_features: torch.Tensor,  # [num_comments, comment_feature_dim] 
                                       stock_comment_edge_index: torch.Tensor) -> HeteroData:
        """
        从特征创建异构图数据
        
        Args:
            stock_features: 股票特征 [num_stocks, 34]
            comment_features: 评论特征 [num_comments, 768]
            stock_comment_edge_index: 股票-评论边索引 [2, num_edges]
        """
        data = HeteroData()
        
        # 添加节点特征
        data['stock'].x = stock_features
        data['comment'].x = comment_features
        
        # 添加边
        data['stock', 'connect', 'comment'].edge_index = stock_comment_edge_index
        
        # 添加反向边
        reverse_edge_index = torch.stack([stock_comment_edge_index[1], stock_comment_edge_index[0]], dim=0)
        data['comment', 'connect_rev', 'stock'].edge_index = reverse_edge_index
        
        return data
    
    def prepare_temporal_sequences(self, 
                                 all_daily_data: List[HeteroData], 
                                 targets: List[Dict[str, torch.Tensor]]) -> Tuple[List[List[HeteroData]], List[Dict[str, torch.Tensor]]]:
        """
        准备时间序列数据和目标值
        
        Args:
            all_daily_data: 所有日期的异构图数据列表
            targets: 每个时间段的目标值列表
            
        Returns:
            sequences: 时间窗口序列列表
            sequence_targets: 对应的目标值列表
        """
        sequences = []
        sequence_targets = []
        
        # 为每个可能的15天窗口创建序列
        for i in range(len(all_daily_data) - self.window_size):
            # 提取15天的序列
            sequence = all_daily_data[i:i + self.window_size]
            sequences.append(sequence)
            
            # 获取对应的目标值
            sequence_targets.append(targets[i + self.window_size])
        
        return sequences, sequence_targets
    
    def calculate_volatility_and_direction(self, 
                                         price_data: np.ndarray, 
                                         horizons: List[int] = [5, 30]) -> Dict[str, float]:
        """
        计算股价波动率和方向
        
        Args:
            price_data: 价格序列 [time_steps, num_stocks]
            horizons: 预测期列表 [5, 30]
            
        Returns:
            包含波动率和方向的字典
        """
        results = {}
        
        for horizon in horizons:
            if len(price_data) >= horizon + 1:
                # 计算收益率
                returns = np.diff(price_data[-(horizon + 1):], axis=0) / price_data[-(horizon + 1):-1]
                
                # 计算波动率 (标准差)
                volatility = np.std(returns, axis=0)  # 对每只股票计算
                results[f'volatility_{horizon}d'] = volatility
                
                # 计算方向 (平均收益率)
                direction = np.mean(returns, axis=0)  # 对每只股票计算
                results[f'direction_{horizon}d'] = direction
            else:
                # 如果数据不足，返回零值
                num_stocks = price_data.shape[1] if len(price_data.shape) > 1 else 1
                results[f'volatility_{horizon}d'] = np.zeros(num_stocks)
                results[f'direction_{horizon}d'] = np.zeros(num_stocks)
        
        return results


class StockDataset(torch.utils.data.Dataset):
    """
    股票预测数据集
    """
    def __init__(self, sequences: List[List[HeteroData]], targets: List[Dict[str, torch.Tensor]]):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        return sequence, target


def create_sample_dataset(num_days: int = 100):
    """
    创建示例数据集
    """
    processor = StockHeteroDataProcessor(window_size=15)
    
    # 创建每日的异构图数据
    daily_data_list = []
    targets_list = []
    
    for day in range(num_days):
        # 创建股票特征 (3只股票, 34维特征)
        stock_features = torch.randn(3, 34)
        
        # 创建评论特征 (每只股票5条评论, 768维特征)
        comment_features = torch.randn(15, 768)
        
        # 创建边索引 (股票与评论的连接)
        # 股票0连接评论0-4, 股票1连接评论5-9, 股票2连接评论10-14
        edge_index = torch.zeros((2, 15), dtype=torch.long)
        for i in range(3):  # 3只股票
            for j in range(5):  # 每只股票5条评论
                edge_index[0, i*5 + j] = i  # 股票ID
                edge_index[1, i*5 + j] = i*5 + j  # 评论ID
        
        # 创建异构图数据
        daily_data = processor.create_hetero_data_from_features(
            stock_features, comment_features, edge_index
        )
        daily_data_list.append(daily_data)
        
        # 创建目标值 (模拟)
        targets = {
            'volatility_5d_target': torch.rand(1, 1),
            'direction_5d_target': torch.randn(1, 1),
            'volatility_30d_target': torch.rand(1, 1), 
            'direction_30d_target': torch.randn(1, 1)
        }
        targets_list.append(targets)
    
    # 准备时间序列数据
    sequences, sequence_targets = processor.prepare_temporal_sequences(daily_data_list, targets_list)
    
    # 创建数据集
    dataset = StockDataset(sequences, sequence_targets)
    
    return dataset


def get_data_loader(dataset, batch_size: int = 1, shuffle: bool = True):
    """
    创建数据加载器
    """
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=lambda batch: list(zip(*batch))  # 自定义批处理函数
    )


if __name__ == "__main__":
    # 创建示例数据集
    dataset = create_sample_dataset(num_days=50)
    print(f"Dataset created with {len(dataset)} sequences")
    
    # 创建数据加载器
    dataloader = get_data_loader(dataset, batch_size=2)
    
    # 测试数据加载
    for i, (sequences, targets) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Sequences length: {len(sequences)}")
        print(f"  First sequence days: {len(sequences[0])}")
        print(f"  Targets keys: {list(targets[0].keys())}")
        if i == 0:  # 只打印第一个批次
            break