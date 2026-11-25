from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#加载日志
ea=EventAccumulator("./runs/sentiment",size_guidance={"scalars":0})
ea.Reload()

def get(tag):
    events=ea.Scalars(tag)
    return [e.step+1 for e in events],[e.value for e in events]

#生成epoch图
fig1,a=plt.subplots(1,2,figsize=(12,4))

for tag,name in [("Loss/train","Train"),("Loss/val","Valid")]:
    u1,u2=get(tag)
    a[0].plot(u1,u2,marker="o",label=name)
a[0].set_xlabel("Epoch")
a[0].set_ylabel("Loss")
a[0].set_title("Loss per Epoch")
a[0].legend()
a[0].grid(True,alpha=0.3)

for tag,name in [("Acc/train","Train"),("Acc/val","Valid")]:
    u1,u2=get(tag)
    a[1].plot(u1,u2,marker="o",label=name)
a[1].set_xlabel("Epoch")
a[1].set_ylabel("Accuracy")
a[1].set_title("Accuracy per Epoch")
a[1].legend()
a[1].grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig("chart_epoch.png",dpi=300,bbox_inches="tight")
plt.close()