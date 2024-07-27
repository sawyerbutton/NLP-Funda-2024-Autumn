import os
import logging
import torch
import torch.nn.functional as F

from utils import metrics

# 配置日志记录的格式和级别
logging.basicConfig(format='%(asctime)s -  %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义训练函数
def train(model, optimizer, train_dataloader, valid_dataloader, args):
    best_f1 = 0
    logger.info('Start Training!')  # 开始训练
    for epoch in range(1, args.epochs + 1):
        model.train()  # 将模型设置为训练模式
        for step, (x, y) in enumerate(train_dataloader):
            x, y = x.to(args.device), y.to(args.device)  # 将数据移至指定设备（如GPU）
            pred = model(x)  # 前向传播，获取预测结果
            loss = F.cross_entropy(pred, y)  # 计算交叉熵损失

            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            if (step + 1) % 200 == 0:
                logger.info(f'|EPOCHS| {epoch:>}/{args.epochs} |STEP| {step + 1:>4}/{len(train_dataloader)} |LOSS| {loss.item():>.4f}')

        avg_loss, accuracy, _, _, f1, _ = evaluate(model, valid_dataloader, args)  # 在验证集上评估模型
        logger.info('-' * 50)
        logger.info(f'|* VALID SET *| |VAL LOSS| {avg_loss:>.4f} |ACC| {accuracy:>.4f} |F1| {f1:>.4f}')
        logger.info('-' * 50)

        if f1 > best_f1:
            best_f1 = f1
            logger.info(f'Saving best model... F1 score is {best_f1:>.4f}')
            if not os.path.isdir(args.model_save_path):
                os.mkdir(args.model_save_path)
            torch.save(model.state_dict(), os.path.join(args.model_save_path, "best.pt"))  # 保存最佳模型
            logger.info('Model saved!')


# 定义评估函数
def evaluate(model, valid_dataloader, args):
    with torch.no_grad():  # 禁用梯度计算
        model.eval()  # 将模型设置为评估模式
        losses, correct = 0, 0
        y_hats, targets = [], []
        for x, y in valid_dataloader:
            x, y = x.to(args.device), y.to(args.device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)  # 计算交叉熵损失
            losses += loss.item()

            y_hat = torch.max(pred, 1)[1]  # 获取预测结果的最大值所在的索引
            y_hats += y_hat.tolist()
            targets += y.tolist()
            correct += (y_hat == y).sum().item()  # 计算正确预测的数量

    avg_loss, accuracy, precision, recall, f1, cm = metrics(valid_dataloader, losses, correct, y_hats, targets)  # 计算评价指标
    return avg_loss, accuracy, precision, recall, f1, cm
