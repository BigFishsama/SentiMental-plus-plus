import torch
import time
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW


encoded=load_from_disk("./data/chnsenti_encoded")
encoded.set_format(type="torch",columns=["input_ids","attention_mask","token_type_ids","labels"])
train_loader=DataLoader(encoded["train"],batch_size=16,shuffle=True)
valid_loader=DataLoader(encoded["validation"],batch_size=32,shuffle=False)

# for batch in train_loader:
#     print("batch keys:",batch.keys())
#     print("input_ids shape:",batch["input_ids"].shape)
#     print("attention_mask shape:",batch["attention_mask"].shape)
#     print("label shape:",batch["label"].shape)
#     break

device="cpu"
if torch.cuda.is_available():
    device="cuda"
model=BertForSequenceClassification.from_pretrained("bert-base-chinese",num_labels=2).to(device)
optimizer=AdamW(model.parameters(),lr=2e-5)
writer=SummaryWriter("./runs/sentiment")

# batch=next(iter(train_loader))
# batch={k:v.to(device) for k,v in batch.items()}
# model.eval()
# with torch.no_grad():
#     outputs=model(**batch)
# print("logits:",outputs.logits.shape)
# print("loss:",outputs.loss)

#验证任何集，返回准确率
def evaluate(loader):  
    model.eval()   #切换评估模式
    total_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}

            outputs=model(**batch)  #计算loss
            total_loss+=outputs.loss.item()*batch["labels"].size(0)   #总loss
            correct+=(outputs.logits.argmax(1)==batch["labels"]).sum().item()   #总acc

            total+=batch["labels"].size(0)   #目前处理的总样本数量

    return total_loss/total,correct/total

epochs=3
cur_step=1

for epoch in range(epochs):
    model.train()
    epoch_loss=0.0
    acc=0
    total=0

    for step,batch in enumerate(train_loader):
        batch={k:v.to(device) for k,v in batch.items()}

        outputs=model(**batch)
        loss=outputs.loss

        optimizer.zero_grad()
        loss.backward()   #反向
        optimizer.step()

        epoch_loss+=loss.item()*batch["labels"].size(0)
        acc+=(outputs.logits.argmax(1)==batch["labels"]).sum().item()

        #记录曲线
        if step%10==0:
            writer.add_scalar("Batch/train_loss",loss.item(),cur_step)
            cur_step+=1

        #控制台查看进度
        if step%100==0:
            print(f"Epoch: {epoch+1} Step: {step}/{len(train_loader)} Loss: {loss.item():.4f}")
        total+=batch["labels"].size(0)

    ave_loss=epoch_loss/total
    ave_acc=acc/total

    #验证集检测
    valid_loss,valid_acc=evaluate(valid_loader)

    print(f"[Epoch {epoch+1}] train_loss={ave_loss:.4f} train_acc={ave_acc:.4f}")
    print(f"valid_loss={valid_loss:.4f} ,valid_acc={valid_acc:.4f}")

    writer.add_scalar("Loss/train",ave_loss,epoch)
    writer.add_scalar("Loss/val",valid_loss,epoch)
    writer.add_scalar("Acc/train",ave_acc,epoch)
    writer.add_scalar("Acc/val",valid_acc,epoch)

model.save_pretrained("./model/bert_senti")
print("模型已保存到 ./model/bert_senti")
writer.close()
print("TensorBoard 日志已写入 ./runs/sentiment")