from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader

ds=load_dataset("lansinuote/ChnSentiCorp")
tok=AutoTokenizer.from_pretrained("bert-base-chinese")

# pretest
# text="原神牛逼"
# testoutput1=tok(text,padding="max_length",max_length=16,truncation=True)
# print(testoutput1["input_ids"])
# print(tok.convert_ids_to_tokens(testoutput1["input_ids"]))


# def getLength(temp1):
#     temp2=tok(temp1["text"])
#     return {"length":[len(i) for i in temp2["input_ids"]]}

# ds_length=ds.map(getLength,batched=True)
# a=np.array(ds_length["train"]["length"])

# for i in [80,85,90,95,96,97,98,99]:
#     print(f"For {i:}% data:{np.percentile(a,i):.2f}")

# output：
# For 80% data:169.00
# For 85% data:180.00
# For 90% data:189.00
# For 95% data:241.00
# For 96% data:276.04 √ -256
# For 97% data:319.00
# For 98% data:392.00
# For 99% data:531.02


def yencode(temp1):
    return tok(temp1["text"],padding="max_length",max_length=256,truncation=True)
new_ds=ds.map(yencode,batched=True)
new_ds=new_ds.rename_column("label","labels")

new_ds.save_to_disk("./data/chnsenti_encoded")
tok.save_pretrained("./model/bert_senti")