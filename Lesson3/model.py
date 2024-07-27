import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, hidden_size_linear, class_num, dropout):
        super(RCNN, self).__init__()
        
        # 嵌入层，将输入的词索引转换为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向LSTM层，用于提取序列信息
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True, dropout=dropout)
        
        # 线性变换层，将LSTM输出与嵌入拼接后的向量进行线性变换
        self.W = nn.Linear(embedding_dim + 2*hidden_size, hidden_size_linear)
        
        # Tanh激活函数
        self.tanh = nn.Tanh()
        
        # 最后的全连接层，将特征向量映射到类别空间
        self.fc = nn.Linear(hidden_size_linear, class_num)

    def forward(self, x):
        # x 的尺寸为 |批次大小, 序列长度|
        
        # 嵌入层，x_emb 的尺寸为 |批次大小, 序列长度, 嵌入维度|
        x_emb = self.embedding(x)
        
        # LSTM层，output 的尺寸为 |批次大小, 序列长度, 2*隐藏层大小|
        output, _ = self.lstm(x_emb)
        
        # 将 LSTM 的输出和原始嵌入拼接，output 的尺寸为 |批次大小, 序列长度, 嵌入维度 + 2*隐藏层大小|
        output = torch.cat([output, x_emb], 2)
        
        # 线性变换和 Tanh 激活，output 的尺寸从 |批次大小, 序列长度, 线性隐藏层大小| 变为 |批次大小, 线性隐藏层大小, 序列长度|
        output = self.tanh(self.W(output)).transpose(1, 2)
        
        # 通过最大池化层，将 output 的尺寸从 |批次大小, 线性隐藏层大小, 序列长度| 变为 |批次大小, 线性隐藏层大小|
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        
        # 最后通过全连接层，output 的尺寸为 |批次大小, 类别数|
        output = self.fc(output)
        
        return output
