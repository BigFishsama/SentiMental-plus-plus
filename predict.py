from transformers import pipeline
import torch

model_path = "./model/bert_senti"
device=-1
if torch.cuda.is_available():
    device=0

sentiment_pipeline = pipeline("sentiment-analysis", model=model_path,tokenizer=model_path,device=device)

texts=["这个游戏好玩","天气真差"]
results=sentiment_pipeline(texts)

for text,result in zip(texts,results):
    print(f"文本: {text}")
    print(f"预测: {result['label']}, 置信度: {result['score']:.4f}\n")