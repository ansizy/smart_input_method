"""
数据预处理
    数据划分    train val
"""
import jieba
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from tokenizer import JiebaTokenizer


# 构建数据集 训练集, 测试集
def build_dataset(sentences, tokenizer):
    # 索引化的 sentences
    indexed_sentences = []
    for sentence in tqdm(sentences, desc="索引化"):
        indexed_sentences.append(tokenizer.encode(sentence))

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
    JiebaTokenizer.build_vocab(train_sentences, config.MODELS_DIR / 'vocab.txt')

    # 6 构建训练集 索引化 --> 构建 inputs targets
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    train_dataset = build_dataset(train_sentences, tokenizer)

    # 7 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR / "train.jsonl", orient='records', lines=True)

    # 8 构建测试集
    test_dataset = build_dataset(test_sentences, tokenizer)

    # 9 保存测试集
    pd.DataFrame(test_dataset).to_json(config.PROCESSED_DATA_DIR / "test.jsonl", orient='records', lines=True)

    print("数据处理完成")


if __name__ == '__main__':
    raw_data_path = config.RAW_DATA_DIR / "synthesized_.jsonl"
    # raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "synthesized_.jsonl"
    process(raw_data_path)
