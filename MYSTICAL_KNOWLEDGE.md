### #0x01 DatasetDict，Dataset

> ```python
> ds = load_dataset("xxx")
> ```

DatasetDict，本质结构类似字典，类似c++的命名空间

- 键：类型为str，“train”对应的是训练集，还有"validation"，"test"
- 值：类型为Dataset

Dataset，本质结构列式表，有两列

- **“text”**列：类型为str，存的是文本信息
- **“label”**列：类型为int，存的是与同行text对应的标签
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



### #0x03 tokenizer 的参数

> ```python
> tok(text, padding="max_length", max_length=256, truncation=True)
> ```

调用分词器时的三个常用参数：

- padding：填充方式，"max_length"表示统一填充到 max_length 的长度，短了补0，长了截断
- max_length：最大长度，配合下面两个参数用
- truncation=True：允许截断，超过 max_length 的部分直接砍掉

（分词器一次调用返回一个字典，就是 #0x02 那三层 input_ids / attention_mask / token_type_ids）



### #0x04 Dataset 的常用方法

> ```python
> new_ds = ds.map(yencode, batched=True)            # 逐批处理，新增列
> new_ds = new_ds.rename_column("label", "labels")  # 重命名列
> new_ds.save_to_disk("./data/chnsenti_encoded")    # 保存到本地磁盘
> ```

- `map(func, batched=True)`：对每一行（或一批）调用 func，func 返回的键值对会变成新的列
- `rename_column(old, new)`：改列名。这里把 "label" 改成 "labels"，因为 BERT 模型的输入字段名就叫 labels
- `save_to_disk(path)`：把处理好的数据集存到本地，下次直接 load，不用重新分词

> ```python
> encoded = load_from_disk("./data/chnsenti_encoded")   # 从本地加载
> encoded.set_format(type="torch", columns=["input_ids","attention_mask","token_type_ids","labels"])
> ```

- `load_from_disk(path)`：加载 save_to_disk 保存的数据集
- `set_format(type="torch", columns=[...])`：指定哪些列转成 torch 张量，好喂给 DataLoader



### #0x05 DataLoader

> ```python
> train_loader = DataLoader(encoded["train"], batch_size=16, shuffle=True)
> valid_loader = DataLoader(encoded["validation"], batch_size=32, shuffle=False)
> ```

DataLoader，数据加载器，把数据集按 batch 分好、一捆一捆喂给模型：

- batch_size：每批多少条样本（训练用16，验证用32）
- shuffle：是否打乱顺序（训练打乱防止过拟合，验证不打乱）

> ```python
> for batch in train_loader:
>     print(batch.keys())   # batch 是个字典，key 就是 set_format 里指定的列名
> ```

遍历得到的每个 batch 是字典，shape 为 [batch_size, max_length]。



### #0x06 模型加载与设备

> ```python
> device = "cpu"
> if torch.cuda.is_available():
>     device = "cuda"
> model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2).to(device)
> ```

- `BertForSequenceClassification`：在预训练 BERT 上加一个分类头，做序列分类
- `from_pretrained(name, num_labels=2)`：加载预训练权重；num_labels=2 表示二分类
- `.to(device)`：把模型搬到指定设备（GPU/CPU）
- `torch.cuda.is_available()`：判断有没有 GPU，有就用 cuda，没有就 cpu



### #0x07 模型的输出

> ```python
> outputs = model(**batch)     # 前向传播
> outputs.loss                 # 损失值
> outputs.logits               # 未归一化的分数
> outputs.logits.argmax(1)     # 取最大值的下标 = 预测类别
> ```

- `model(**batch)`：batch 是字典，`**batch` 解包成关键字参数传给模型
- `outputs.loss`：模型自己算好的交叉熵损失
- `outputs.logits`：形状 [batch_size, num_labels]，每行是每个类别的得分
- `.argmax(1)`：按第1维（列）取最大值的下标，得到预测标签



### #0x08 训练三件套 + 模式切换

> ```python
> optimizer.zero_grad()   # 1. 清空梯度
> loss.backward()         # 2. 反向传播，算梯度
> optimizer.step()        # 3. 更新参数
> ```

- zero_grad()：把上一次的梯度清零，否则梯度会累加
- backward()：反向传播，计算每个参数的梯度
- step()：根据梯度和学习率更新参数

> ```python
> model.train()          # 训练模式（打开 dropout 等）
> model.eval()           # 评估模式（关闭 dropout）
> with torch.no_grad():  # 不记录梯度，省显存、加速（评估时用）
> ```



### #0x09 AdamW 优化器

> ```python
> optimizer = AdamW(model.parameters(), lr=2e-5)
> ```

- AdamW：带权重衰减的 Adam 优化器，BERT 微调的标配
- model.parameters()：模型所有可训练参数
- lr：学习率。BERT 微调常用 2e-5（很小，因为预训练权重已经很好了，步子不能太大）



### #0x0A TensorBoard 日志

> ```python
> writer = SummaryWriter("./runs/sentiment")          # 写日志
> writer.add_scalar("Acc/val", valid_acc, epoch)      # 记录一个标量
> writer.close()                                       # 关闭
> ```

- SummaryWriter(path)：创建写日志对象，数据写到指定目录
- add_scalar(tag, value, step)：记录标量，tag 是名字（如 "Acc/val"），step 是横坐标（epoch）

> ```python
> ea = EventAccumulator("./runs/sentiment", size_guidance={"scalars":0})  # 读日志
> ea.Reload()                # 加载
> ea.Scalars("Acc/val")      # 取某个 tag 的所有 (step, value)
> ```

- EventAccumulator：反过来读日志，Scalars(tag) 返回 [(step, value), ...] 的列表



### #0x0B matplotlib 绘图

> ```python
> fig, a = plt.subplots(1, 2, figsize=(12, 4))    # 1行2列子图
> a[0].plot(x, y, marker="o", label="Train")      # 折线图
> a[0].set_xlabel("Epoch")                         # x轴标签
> a[0].set_ylabel("Loss")                          # y轴标签
> a[0].set_title("Loss per Epoch")                 # 标题
> a[0].legend()                                    # 图例
> a[0].grid(True, alpha=0.3)                       # 网格线
> plt.tight_layout()
> plt.savefig("chart_epoch.png", dpi=300, bbox_inches="tight")
> ```

- subplots(1, 2, figsize=(w, h))：创建子图，figsize 是画布尺寸（英寸）
- plot(x, y, marker, label)：画折线，label 给图例用
- savefig(path, dpi, bbox_inches)：保存图片，dpi=300 高清，bbox_inches="tight" 裁掉多余空白



### #0x0C 一些小工具

> ```python
> np.percentile(a, 96)                              # 第96百分位的值
> glob.glob("./runs/sentiment/events.out.tfevents.*")  # 通配符找文件
> max(files, key=os.path.getmtime)                  # 按修改时间取最新的
> np.convolve(v, np.ones(10)/10, mode="valid")      # 10点滑动平均
> ```

- np.percentile(a, i)：算第 i 百分位的值（用来统计 token 长度分布，决定 max_length=256）
- glob.glob(pattern)：用通配符匹配文件路径，返回列表
- max(files, key=os.path.getmtime)：按文件修改时间排序，取最新的那个（解决日志混杂那个坑）
- np.convolve(v, kernel, "valid")：卷积，用全 1/窗口 做滑动平均，平滑曲线



### #0x0D pipeline 推理

> ```python
> sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, device=device)
> results = sentiment_pipeline(texts)
> # results = [{"label": "LABEL_0", "score": 0.99}, ...]
> ```

- pipeline(task, model, tokenizer, device)：一键推理，task 指定任务（sentiment-analysis = 情感分析）
- 传入一段文本（或列表），返回 [{label, score}]，label 是标签，score 是置信度
- device：-1 用 CPU，0 用第一块 GPU
