# smart_input_method
智能输入法，通过普通 ```rnn``` 网络做一款输入法
## 整体目录结构
```python
"""
+---data    # 存放数据
|   +---processed   # 经过处理的数据
|   |       test.jsonl
|   |       train.jsonl
|   |
|   \---raw # 原始数据
|           synthesized_.jsonl
|
+---logs    # tensorboard logs
+---models
|       best.pt # 模型
|       vocab.txt   # 词表
|
+---src
|   |   config.py   # 配置文件
|   |   dataset.py  # 创建数据集
|   |   evaluate.py # 评估
|   |   model.py    # 定义模型
|   |   predict.py  # 预测脚本
|   |   process.py  # 数据预处理
|   |   tokenizer.py    # tokenizer
|   |   train.py    # 训练
|
\---test    # 一些测试
    |   test_tensorboard.py
    |
    \---logs
            events.out.tfevents.1774182503.DESKTOP-T0CONCQ.20208.0
"""
```
