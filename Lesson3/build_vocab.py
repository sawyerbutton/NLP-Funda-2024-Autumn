from collections import Counter  # 从collections模块中导入Counter类

def build_dictionary(texts, vocab_size):
    counter = Counter()  # 创建Counter对象用于统计单词频率
    SPECIAL_TOKENS = ['<PAD>', '<UNK>']  # 定义特殊标记：填充符和未知词

    for word in texts:
        counter.update(word)  # 更新计数器，统计每个单词的出现频率

    # 获取出现频率最高的单词，并与特殊标记合并形成最终词汇表
    words = [word for word, count in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
    words = SPECIAL_TOKENS + words

    # 创建单词到索引的映射
    word2idx = {word: idx for idx, word in enumerate(words)}

    return word2idx  # 返回单词到索引的映射
