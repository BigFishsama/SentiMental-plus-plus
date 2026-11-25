from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

#日志混杂历史遗留训练数据，所以需要根据日期找最近的
event_files=glob.glob(os.path.join("./runs/sentiment","events.out.tfevents.*"))
latest_file=max(event_files,key=os.path.getmtime)

#加载日志
ea=EventAccumulator(latest_file,size_guidance={"scalars":0})
ea.Reload()

def get(tag):
    events=ea.Scalars(tag)
    return [e.step+1 for e in events],[e.value for e in events]

#生成batch图
fig2,b=plt.subplots(figsize=(8,4))
s,v=get("Batch/train_loss")
b.plot(s,v,alpha=0.4,label="Raw")

#test
# print("数据点总数:", len(s))
# print("s1:",s[:10])
# print("s2:",s[-10:])
# print("v1:",v[:10])
# print("v2:",v[-10:])

#平滑绘图
window=10
if len(v)>=window:
    sm=np.convolve(v,np.ones(window)/window, mode="valid")
    b.plot(s[window-1:],sm,color="C1",label=f"Smoothed (w={window})")

b.set_xlabel("Step(x10)")
b.set_ylabel("Training Loss")
b.set_title("Batch Training Loss")
b.legend()
b.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig("chart_batch.png",dpi=300,bbox_inches="tight")
plt.close()