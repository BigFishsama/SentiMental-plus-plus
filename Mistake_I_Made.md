# 该项目遇到的一系列问题

> *made by fishsoup -2026.11.21*

### #0x01 忘记切换虚拟环境

创建虚拟环境：

>```powershell
>python -m venv nlp_env
>```

**终端切换虚拟环境**：

> ```powershell
> D:\C++pro\nlp_env\Scripts\Activate.ps1
> ```

> ```powershell
> pip install torch torchvision torchaudio
> pip install transformers
> pip install datasets
> ```

### #0x02 .gitignore文件编码出错

该项目第一次push的时候，发现莫名其妙的 **`.gitignore` 文件失效**。

这个错误让人非常困惑，以至于翻遍全网博客没有找到解决方法，后来经过排查发现，由于我的 `.gitignore` 文件并不是项目创建初自带的，而是通过本地 powershell 直接创建，导致编码为 UTF-16，**正确编码应该是 UTF-8**，直接在vscode中切换编码即可。

> ```bash
> /nlp_env/
> /huggingface_cache/
> *.pyc
> 
> #防止上传虚拟环境，huggingface缓存，以及py运行文件
> ```

### #0x03 git大文件传输错误

在微调好模型之后，push项目的时候又双叒叕出问题了，**这次因为 `.safetensors` 文件超过 100兆，不能通过git上传。**

这时候需要通过 `git lfs`，一个专门为大文件上传的工具。

> ```powershell
> git lfs install
> git lfs track "*.safetensors"
> ```

在前文 `.gitignore` 文件出错的时候，需要在重新提交前把 git工作区的缓存清空，重新跟踪文件。

> ```
> git rm --cached ...
> ```

匪夷所思的地方来了，即便清空缓存 `git lfs` 依旧无法生效，忍无可忍之下，直接彻底把仓库历史记录彻底清理了

> ```powershell
> pip install git-filter-repo  #一个抹除文件在仓库历史记录的工具
> git filter-repo --path model/bert_senti/model.safetensors --invert-paths
> ```

上述两个问题都是**由于没有提前规划好项目所需要的工具**，如果提前配置好追踪文件就不会有问题。
