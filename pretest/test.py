import torch
from transformers import pipeline

print("当前PyTorch版本:",torch.__version__)
print("CUDA是否可用:",torch.cuda.is_available())

classifier=pipeline("sentiment-analysis")
result=classifier("I love you")
print(result)