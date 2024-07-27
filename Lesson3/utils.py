import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
nltk.download('punkt')  # 下载nltk中的punkt模型，用于分词

def read_file(file_path):
    """
    读取AG NEWS数据集的函数
    """
    data = pd.read_csv(file_path, names=["class", "title", "description"])  # 读取CSV文件，并指定列名
    texts = list(data['title'].values + ' ' + data['description'].values)  # 将标题和描述连接成一个文本
    texts = [word_tokenize(preprocess_text(sentence)) for sentence in texts]  # 对每个文本进行预处理和分词
    labels = [label-1 for label in list(data['class'].values)]  # 标签从1~4转为0~3
    return texts, labels  # 返回处理后的文本和标签

def preprocess_text(string):
    """
    预处理文本的函数
    """
    string = string.lower()  # 转换为小写
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  # 移除特定字符，保留字母数字及某些标点
    string = re.sub(r"\'s", " \'s", string)  # 处理所有者's
    string = re.sub(r"\'ve", " \'ve", string)  # 处理've
    string = re.sub(r"n\'t", " n\'t", string)  # 处理否定的n't
    string = re.sub(r"\'re", " \'re", string)  # 处理're
    string = re.sub(r"\'d", " \'d", string)  # 处理'd
    string = re.sub(r"\'ll", " \'ll", string)  # 处理'll
    string = re.sub(r",", " , ", string)  # 处理逗号
    string = re.sub(r"!", " ! ", string)  # 处理感叹号
    string = re.sub(r"\(", " \( ", string)  # 处理左括号
    string = re.sub(r"\)", " \) ", string)  # 处理右括号
    string = re.sub(r"\?", " \? ", string)  # 处理问号
    string = re.sub(r"\s{2,}", " ", string)  # 将多个空格替换为一个空格
    return string.strip()  # 去除字符串两端的空格

def metrics(dataloader, losses, correct, y_hats, targets):
    """
    计算模型性能指标的函数
    """
    avg_loss = losses / len(dataloader)  # 计算平均损失
    accuracy = correct / len(dataloader.dataset) * 100  # 计算准确率
    precision = precision_score(targets, y_hats, average='macro')  # 计算精确率
    recall = recall_score(targets, y_hats, average='macro')  # 计算召回率
    f1 = f1_score(targets, y_hats, average='macro')  # 计算F1分数
    cm = confusion_matrix(targets, y_hats)  # 计算混淆矩阵
    return avg_loss, accuracy, precision, recall, f1, cm  # 返回所有计算的指标
