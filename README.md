# GPT2_FROM_SCRATCH
从头开始实现一个GPT2

## 环境（4090）
- Python：3.9
- CUDA：11.8
- Pytorch：2.0.1+cu118

### 2024/10/29：首次上传，实现从头训练，差微调代码
- main.py：训练代码
- dataset.py：数据集相关代码
- config.py：模型配置文件
- attention.py：注意力模块
- gpt2.py：GPT2建模文件
- utils.py：相关工具函数
- gpt_download.py：预训练参数下载

### 2024/10/30：更新分类微调
- main_finetune_classifier.py：基于GPT2预训练结果进行垃圾邮件分类任务微调

### 2024/11/02：更新指令微调
- main_finetune_instruct.py：基于GPT2-355M的指令微调任务

## 备注
- 指令微调包含利用其他大模型进行评分，需要下载Ollama进行API调用。Ollama下载模型会超时，采用离线部署方式（https://bbs.fit2cloud.com/t/topic/4628）