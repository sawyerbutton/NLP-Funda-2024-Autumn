import os  # 导入操作系统接口模块
import argparse  # 导入用于解析命令行参数的模块
import logging  # 导入日志模块
import random  # 导入随机数模块
import numpy as np  # 导入NumPy库，用于科学计算
import torch  # 导入PyTorch库
from torch.utils.data import DataLoader, random_split  # 导入PyTorch的数据加载器和数据集拆分功能

from build_vocab import build_dictionary  # 从build_vocab模块中导入build_dictionary函数
from dataset import CustomTextDataset, collate_fn  # 从dataset模块中导入CustomTextDataset类和collate_fn函数
from model import RCNN  # 从model模块中导入RCNN类
from trainer import train, evaluate  # 从trainer模块中导入train和evaluate函数
from utils import read_file  # 从utils模块中导入read_file函数

logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)  # 配置日志格式和级别
logger = logging.getLogger(__name__)  # 创建一个日志记录器对象

def set_seed(args):
    random.seed(args.seed)  # 设置Python随机数生成器的种子
    np.random.seed(args.seed)  # 设置NumPy随机数生成器的种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机数生成器的种子
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 如果有多个GPU设备，设置所有GPU的随机数生成器种子

def main(args):
    model = RCNN(vocab_size=args.vocab_size,
                 embedding_dim=args.embedding_dim,
                 hidden_size=args.hidden_size,
                 hidden_size_linear=args.hidden_size_linear,
                 class_num=args.class_num,
                 dropout=args.dropout).to(args.device)  # 初始化RCNN模型并将其移动到指定设备

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, dim=0)  # 如果有多个GPU设备，使用DataParallel进行并行计算

    train_texts, train_labels = read_file(args.train_file_path)  # 读取训练数据文件
    word2idx = build_dictionary(train_texts, vocab_size=args.vocab_size)  # 构建词汇表
    logger.info('Dictionary Finished!')  # 记录词汇表构建完成的日志信息

    full_dataset = CustomTextDataset(train_texts, train_labels, word2idx)  # 创建自定义文本数据集
    num_train_data = len(full_dataset) - args.num_val_data  # 计算训练数据的数量
    train_dataset, val_dataset = random_split(full_dataset, [num_train_data, args.num_val_data])  # 随机拆分数据集为训练集和验证集
    train_dataloader = DataLoader(dataset=train_dataset,
                                  collate_fn=lambda x: collate_fn(x, args),
                                  batch_size=args.batch_size,
                                  shuffle=True)  # 创建训练数据加载器

    valid_dataloader = DataLoader(dataset=val_dataset,
                                  collate_fn=lambda x: collate_fn(x, args),
                                  batch_size=args.batch_size,
                                  shuffle=True)  # 创建验证数据加载器

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 初始化Adam优化器
    train(model, optimizer, train_dataloader, valid_dataloader, args)  # 调用训练函数
    logger.info('******************** Train Finished ********************')  # 记录训练完成的日志信息

    # 测试
    if args.test_set:
        test_texts, test_labels = read_file(args.test_file_path)  # 读取测试数据文件
        test_dataset = CustomTextDataset(test_texts, test_labels, word2idx)  # 创建自定义测试数据集
        test_dataloader = DataLoader(dataset=test_dataset,
                                     collate_fn=lambda x: collate_fn(x, args),
                                     batch_size=args.batch_size,
                                     shuffle=True)  # 创建测试数据加载器

        model.load_state_dict(torch.load(os.path.join(args.model_save_path, "best.pt")))  # 加载最优模型参数
        _, accuracy, precision, recall, f1, cm = evaluate(model, test_dataloader, args)  # 调用评估函数并获取评估指标
        logger.info('-'*50)
        logger.info(f'|* TEST SET *| |ACC| {accuracy:>.4f} |PRECISION| {precision:>.4f} |RECALL| {recall:>.4f} |F1| {f1:>.4f}')  # 记录测试集的评估结果
        logger.info('-'*50)
        logger.info('---------------- CONFUSION MATRIX ----------------')  # 记录混淆矩阵信息
        for i in range(len(cm)):
            logger.info(cm[i])
        logger.info('--------------------------------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--seed', type=int, default=42)  # 添加随机种子参数
    parser.add_argument('--test_set', action='store_true', default=False)  # 添加测试集参数

    # 数据相关参数
    parser.add_argument("--train_file_path", type=str, default="./data/train.csv")  # 训练数据文件路径
    parser.add_argument("--test_file_path", type=str, default="./data/test.csv")  # 测试数据文件路径
    parser.add_argument("--model_save_path", type=str, default="./model_saved")  # 模型保存路径
    parser.add_argument("--num_val_data", type=int, default=10000)  # 验证集数据数量
    parser.add_argument("--max_len", type=int, default=64)  # 最大序列长度
    parser.add_argument("--batch_size", type=int, default=64)  # 批处理大小

    # 模型相关参数
    parser.add_argument("--vocab_size", type=int, default=8000)  # 词汇表大小
    parser.add_argument("--embedding_dim", type=int, default=300)  # 词向量维度
    parser.add_argument("--hidden_size", type=int, default=512)  # 隐藏层大小
    parser.add_argument("--hidden_size_linear", type=int, default=512)  # 线性层隐藏层大小
    parser.add_argument("--class_num", type=int, default=4)  # 类别数量
    parser.add_argument("--dropout", type=float, default=0.0)  # Dropout概率

    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=10)  # 训练轮数
    parser.add_argument("--lr", type=float, default=3e-4)  # 学习率
    args = parser.parse_args()  # 解析命令行参数

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU
    args.n_gpu = torch.cuda.device_count()  # 获取可用GPU的数量
    set_seed(args)  # 设置随机种子

    main(args)  # 调用主函数
