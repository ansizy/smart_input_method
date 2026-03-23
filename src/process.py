"""
数据预处理
    数据划分    train val
"""
import jieba
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config

# 构建数据集 训练集, 测试集
def build_dataset(word2index, sentences):
    # 索引化的 sentences
    indexed_sentences = []
    for sentence in tqdm(sentences, desc="索引化"):
        words = jieba.lcut(sentence)
        indexed_sentence = []
        for word in words:
            if word in word2index:
                indexed_sentence.append(word2index[word])
            else:
                indexed_sentence.append(word2index['unk'])
        indexed_sentences.append(indexed_sentence)

    # 构建 inputs targets
    windows_size = 5
    # dataset: [{"input": [1, 2, 3, 4, 5], "target": 6}, ....]
    dataset = []
    for indexed_sentence in tqdm(indexed_sentences, desc="构建数据集"):
        p = windows_size
        while p < len(indexed_sentence):
            input = indexed_sentence[p - windows_size:p]
            target = indexed_sentence[p]
            dataset.append({"input": input, "target": target})
            p += 1

    return dataset

def process(raw_data_path):
    # 1 读文件
    df = pd.read_json(raw_data_path, orient='records', lines=True).sample(frac=0.1)

    # 2 提取句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])
    # print(len(sentences))
    # print(sentences[0:10])

    # 3 划分数据集
    train_sentences, test_sentences = train_test_split(sentences, train_size=0.8)

    # 4 构建词表
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc="构建词表"):
        # 切词并加入集合去重
        vocab_set.update(jieba.lcut(sentence))

    # 加入 未知字符 'unk'
    vocab_list = ['unk'] + list(vocab_set)
    # print(len(vocab_list))
    # 5 保存词表
    with open(config.MODELS_DIR / 'vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    # 6 构建训练集 索引化 --> 构建 inputs targets
    word2index = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index

    train_dataset = build_dataset(word2index, train_sentences)

    # 7 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / "train.jsonl", orient='records', lines=True)

    # 8 构建测试集
    test_dataset = build_dataset(word2index, test_sentences)

    # 9 保存测试集
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / "test.jsonl", orient='records', lines=True)

    print("数据处理完成")









if __name__ == '__main__':
    raw_data_path = config.RAW_DATA_DIR / "synthesized_.jsonl"
    # raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "synthesized_.jsonl"
    process(raw_data_path)