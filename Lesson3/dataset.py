import torch
from torch.utils.data import Dataset

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, dictionary):
        # 未知的词汇对应的索引是1 (<UNK>)
        self.x = [[dictionary.get(token, 1) for token in token_list] for token_list in texts]
        self.y = labels

    def __len__(self):
        """返回数据集的长度"""
        return len(self.x)

    def __getitem__(self, idx):
        """返回给定索引处的一条数据"""
        return self.x[idx], self.y[idx]

def collate_fn(data, args, pad_idx=0):
    """填充"""
    texts, labels = zip(*data)
    texts = [s + [pad_idx] * (args.max_len - len(s)) if len(s) < args.max_len else s[:args.max_len] for s in texts]
    return torch.LongTensor(texts), torch.LongTensor(labels)
