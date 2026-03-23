import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import InputMethodModel
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    top1_acc_count = 0
    top5_acc_count = 0
    total_count = 0

    for inputs, targets in tqdm(test_dataloader, desc="Evaluating"):
        top5_indexed_list = predict_batch(model, inputs, device)
        # top5_indexed_list shape [batch_size, 5]
        targets.tolist()
        # targets shape [batch_size]

        for index ,top5_indexed in enumerate(top5_indexed_list):
            total_count += 1
            if targets[index] == top5_indexed[0]:
                top1_acc_count += 1
            if targets[index] in top5_indexed:
                top5_acc_count += 1
    return top1_acc_count / total_count, top5_acc_count / total_count


def run_evaluate():
    # 准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")

    # 加载模型
    model = InputMethodModel(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pt", map_location=device))

    # test数据集
    test_dataloader = get_dataloader(train=False)

    # 评估
    top1_acc, top5_acc = evaluate(model, test_dataloader, device)
    print("评估结果")
    print("Top-1 Accuracy: ", top1_acc)
    print("Top-5 Accuracy: ", top5_acc)


if __name__ == '__main__':
    run_evaluate()