# from transformers import BertTokenizer, BertModel
#
# tz = BertTokenizer.from_pretrained("bert-base-chinese")
# bt= BertModel.from_pretrained("bert-base-chinese")
#
# # 返回bert切成token之后的结果
# tz.tokenize("今天的天气怎么样")
# # ['今', '天', '的', '天', '气', '怎', '么', '样']
# input=tz.tokenize("今天的天气怎么样")
#
#
#
#
# outputs=bt(**input)
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)
#
#
# # 将token转化成对应的id，如果token不存在，返回未登录词的token 100
# print(tz.convert_tokens_to_ids(tz.tokenize("今天的天气怎么样")))
# # [791, 1921, 4638, 1921, 3698, 2582, 720, 3416]


import torch
from transformers import BertTokenizer, BertModel

# 函数from_pretrained()根据相应配置信息实例化一个预训练模型
bert = BertModel.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = bert(**inputs)

last_hidden_states = outputs.last_hidden_state