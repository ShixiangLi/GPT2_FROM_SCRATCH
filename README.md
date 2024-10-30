# GPT2_FROM_SCRATCH
从头开始实现一个GPT2

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