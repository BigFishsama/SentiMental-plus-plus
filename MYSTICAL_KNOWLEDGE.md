### #0x01 DatasetDict，Dataset

> ```python
> ds = load_dataset("xxx")
> ```

DatasetDict，本质结构类似字典，类似c++的命名空间

- 键：类型为str，“train”对应的是训练集，还有"validation"，"test"
- 值：类型为Dataset

Dataset，本质结构列式表，有两列

- **“test”**列：类型为str，存的是文本信息
- **“label”**列：类型为int，存的是与同行test对应的标签
- 也可以通过下标访问行。

> ```python
> len(ds["xx"])   #获取"xx"集行数
> ds.map(factory_func,batched=True)
> #对每一项（行）导入工厂函数中处理，得到一个新的列
> #参数的类型是Dataset，返回值类型是多个或单个键值对
> ```



### #0x02 AutoTokenizer

> ```python
> tok=AutoTokenizer.from_pretrained("bert-base-chinese")
> ```

tokenizer，分词器，通俗的来说可以将自然语言转化为token，具体结构分为三层：

- input_ids：对应每个token的词表id，根据调用的模型不同会有所变动，但固定的几个符号不会改变id
- attention_mask：值仅为0/1，如果采用填充的话，可以用于区分真假token
- token_type_ids：值仅为0/1，用于区分第一句和第二句话









