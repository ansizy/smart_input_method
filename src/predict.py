import jieba
import torch

import config
from model import InputMethodModel

def predict_batch(model, inputs, device):
    """
    批量预测
    :param model: 模型
    :param inputs: 输入 shape[batch_size, seq_len]
    :param device: cuda or cpu
    :return: 预测结果 list shape: [batch_size, 5]
    """
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(inputs.to(device))
        # outputs shape [batch_size, vocab_size)
        top5_indexes = torch.topk(outputs, 5).indices
        # top5_indexes shape [batch_size, 5]
        top5_indexes_list = top5_indexes.tolist()
    return top5_indexes_list

def predict(text, model, device, word2index, index2word):

    # 分词
    tokens = jieba.lcut(text)
    indexes = []
    for token in tokens:
        if token in word2index:
            indexes.append(word2index[token])
        else:
            indexes.append(word2index["unk"])
    input_tensor = torch.tensor(indexes, dtype=torch.long).reshape(1, -1)
    top5_indexes_list = predict_batch(model, input_tensor, device)
    top5_tokens = []
    for index in top5_indexes_list[0]:
        top5_tokens.append(index2word[index])
    return top5_tokens

def run_predict():

    # 准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载词表
    vocab_list = []
    with open(config.MODELS_DIR / "vocab.txt", "r", encoding="utf-8") as f:
        for line in f:
            vocab_list.append(line.strip())
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}

    # 加载模型
    model = InputMethodModel(len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / "best.pt", map_location=device))


    print("欢迎使用smart_input_model(输入 q或quit 退出)")
    input_history = ''
    while True:
        user_input = input(">")
        if user_input in ["q", "quit"]:
            print("退出!")
            break
        elif user_input.strip() == '':
            print("请输入内容")
            continue
        else:
            input_history += user_input
            print(f"历史输入: {input_history}")
            top5_tokens = predict(input_history, model, device, word2index, index2word)
            print(top5_tokens)


if __name__ == '__main__':
    run_predict()