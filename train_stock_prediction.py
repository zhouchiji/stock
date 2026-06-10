import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.data import HeteroData
import numpy as np
from typing import Dict, List, Tuple
from data_processor import StockHeteroDataProcessor, StockDataset, get_data_loader, create_sample_dataset
import os


class HeteroGNNBlock(nn.Module):
    """
    异构图神经网络块，处理股票和评论之间的异构关系
    """
    def __init__(self, in_channels: Dict[str, int], hidden_channels: int, out_channels: int, 
                 metadata, dropout: float = 0.1):
        super().__init__()
        self.convs = torch.nn.ModuleDict()
        
        # 定义异构卷积层
        conv_dict = {}
        for edge_type in metadata[1]:  # metadata[1] 包含边类型
            src, relation, dst = edge_type
            conv_dict[edge_type] = SAGEConv(
                in_channels=in_channels[src], 
                out_channels=hidden_channels
            )
        
        self.conv = HeteroConv(conv_dict, aggr='mean')
        
        # 节点类型特定的线性层
        self.node_linears = nn.ModuleDict()
        for node_type in metadata[0]:  # metadata[0] 包含节点类型
            self.node_linears[node_type] = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_dict, edge_index_dict):
        # 应用异构卷积
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # 应用节点类型特定的线性层
        for node_type, linear in self.node_linears.items():
            if node_type in x_dict:
                x_dict[node_type] = linear(x_dict[node_type])
        
        return x_dict


class StockHeteroGNN(nn.Module):
    """
    股票预测的异构图神经网络主模型
    """
    def __init__(self, stock_features: int = 34, comment_features: int = 768, 
                 hidden_dim: int = 256, num_layers: int = 3, sequence_length: int = 15):
        super().__init__()
        
        self.stock_features = stock_features
        self.comment_features = comment_features
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # 节点嵌入层
        self.stock_embedding = nn.Linear(stock_features, hidden_dim)
        self.comment_embedding = nn.Linear(comment_features, hidden_dim)
        
        # 异构GNN层
        self.hetero_gnn_layers = nn.ModuleList()
        
        # 第一层：输入维度到隐藏维度
        in_channels = {'stock': hidden_dim, 'comment': hidden_dim}
        self.hetero_gnn_layers.append(
            HeteroGNNBlock(
                in_channels=in_channels,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(['stock', 'comment'], [('stock', 'connect', 'comment'), ('comment', 'connect_rev', 'stock')]),
                dropout=0.1
            )
        )
        
        # 后续层
        for _ in range(num_layers - 1):
            self.hetero_gnn_layers.append(
                HeteroGNNBlock(
                    in_channels={'stock': hidden_dim, 'comment': hidden_dim},
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(['stock', 'comment'], [('stock', 'connect', 'comment'), ('comment', 'connect_rev', 'stock')]),
                    dropout=0.1
                )
            )
        
        # 序列处理层 - 使用LSTM处理时间序列
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                           num_layers=2, batch_first=True, dropout=0.1)
        
        # 预测头 - 分别预测5日和30日的波动率和方向
        self.volatility_5d_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 波动率通常在[0,1]之间
        )
        
        self.direction_5d_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # 方向在[-1,1]之间
        )
        
        self.volatility_30d_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.direction_30d_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data_list: List[HeteroData]):
        """
        前向传播
        
        Args:
            data_list: 包含15天数据的HeteroData列表
        """
        batch_size = len(data_list[0]['stock'].x) if len(data_list) > 0 else 1
        sequence_features = []
        
        # 处理每个时间步的数据
        for day_data in data_list:
            # 提取节点特征
            x_dict = {
                'stock': day_data['stock'].x,
                'comment': day_data['comment'].x
            }
            
            # 应用嵌入层
            x_dict['stock'] = self.stock_embedding(x_dict['stock'])
            x_dict['comment'] = self.comment_embedding(x_dict['comment'])
            
            # 应用异构GNN层
            for gnn_layer in self.hetero_gnn_layers:
                x_dict = gnn_layer(x_dict, day_data.edge_index_dict)
            
            # 聚合股票节点特征（例如通过平均）
            stock_features = x_dict['stock']
            aggregated_features = torch.mean(stock_features, dim=0, keepdim=True)  # [1, hidden_dim]
            sequence_features.append(aggregated_features)
        
        # 堆叠序列特征 [batch_size, seq_len, hidden_dim]
        seq_tensor = torch.stack(sequence_features, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        # 通过LSTM处理时间序列
        lstm_out, (hidden, cell) = self.lstm(seq_tensor)
        
        # 使用最后一个时间步的输出进行预测
        final_features = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 预测5日波动率和方向
        volatility_5d = self.volatility_5d_head(final_features)
        direction_5d = self.direction_5d_head(final_features)
        
        # 预测30日波动率和方向
        volatility_30d = self.volatility_30d_head(final_features)
        direction_30d = self.direction_30d_head(final_features)
        
        return {
            'volatility_5d': volatility_5d,
            'direction_5d': direction_5d,
            'volatility_30d': volatility_30d,
            'direction_30d': direction_30d
        }


class StockPredictionTrainer:
    """
    训练和评估股票预测模型
    """
    def __init__(self, model: StockHeteroGNN, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数 - 分别为波动率和方向定义
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def compute_loss(self, predictions: Dict, targets: Dict):
        """
        计算总损失
        """
        total_loss = 0
        
        # 波动率预测损失 (使用MSE)
        if 'volatility_5d_target' in targets:
            vol_5d_loss = self.mse_loss(predictions['volatility_5d'], targets['volatility_5d_target'])
            total_loss += vol_5d_loss
        
        if 'volatility_30d_target' in targets:
            vol_30d_loss = self.mse_loss(predictions['volatility_30d'], targets['volatility_30d_target'])
            total_loss += vol_30d_loss
        
        # 方向预测损失 (使用MSE，因为方向是连续值)
        if 'direction_5d_target' in targets:
            dir_5d_loss = self.mse_loss(predictions['direction_5d'], targets['direction_5d_target'])
            total_loss += dir_5d_loss
            
        if 'direction_30d_target' in targets:
            dir_30d_loss = self.mse_loss(predictions['direction_30d'], targets['direction_30d_target'])
            total_loss += dir_30d_loss
        
        return total_loss
    
    def train_epoch(self, dataloader):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (batch_data, targets) in enumerate(dataloader):
            # 将数据移到设备上
            batch_data = [[data.to(self.device) for data in seq] for seq in batch_data][0]  # 简化处理
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets][0]
            
            # 前向传播
            predictions = self.model(batch_data)
            
            # 计算损失
            loss = self.compute_loss(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        """
        评估模型
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, targets in dataloader:
                # 将数据移到设备上
                batch_data = [[data.to(self.device) for data in seq] for seq in batch_data][0]  # 简化处理
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets][0]
                
                predictions = self.model(batch_data)
                loss = self.compute_loss(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.scheduler.step(avg_loss)
        
        return avg_loss


def main():
    print("开始股票预测模型训练...")
    
    # 创建数据集
    print("创建数据集...")
    dataset = create_sample_dataset(num_days=50)
    print(f"数据集创建完成，包含 {len(dataset)} 个序列")
    
    # 创建数据加载器
    dataloader = get_data_loader(dataset, batch_size=1, shuffle=True)
    
    # 创建模型
    print("创建模型...")
    model = StockHeteroGNN(
        stock_features=34, 
        comment_features=768, 
        hidden_dim=128,  # 减小维度以适应示例
        num_layers=2,
        sequence_length=15
    )
    
    # 创建训练器
    trainer = StockPredictionTrainer(model)
    
    # 训练模型
    print("开始训练...")
    for epoch in range(20):  # 减少epoch数以适应示例
        print(f"Epoch {epoch+1}/20")
        
        train_loss = trainer.train_epoch(dataloader)
        eval_loss = trainer.evaluate(dataloader)
        
        print(f"Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
    
    print("训练完成！")
    
    # 保存模型
    model_path = "stock_prediction_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")


if __name__ == "__main__":
    main()