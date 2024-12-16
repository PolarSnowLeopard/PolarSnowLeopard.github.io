---
title: 在CFFF云平台使用llama-factory部署及微调Qwen2.5-7B-Instruct   
date: 2024-12-16 16:00:00 +0800
categories: [llama-factory, CFFF, Qwen, LLM]
tags: [llama-factory, CFFF, Qwen, LLM]     # TAG names should always be lowercase
author: PolarSnowLeopard
---

> - [LLaMA-Factory/README_zh.md at main · hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)
> - [在autodl平台使用llama-factory微调Qwen1.5-7B_autodl部署llama-factory-CSDN博客](https://blog.csdn.net/yidao0618/article/details/138380934)
> - [LLM-2：LLama-factory windows部署及在qwen2-1.5B上的使用_llama-factory qwen2.5 1.5b-CSDN博客](https://blog.csdn.net/m0_61933618/article/details/141856570)
> - [llama-factory 系列教程 (七)，Qwen2.5-7B-Instruct 模型微调与vllm部署详细流程实战_微调qwen2.5后使用vllm运行-CSDN博客](https://blog.csdn.net/sjxgghg/article/details/144016723)

## 1. 部署

### 1. 创建实例

24GB的显存基本可以满足7B模型的部署和微调，不过由于CFFF平台都是A100显卡，因此选择一张GPU创建云服务器实例（AI4S_share_queue, A100 * 1, 80G）



### 2. 安装环境

首先参照官方文档拉取并安装LLaMA-Factory，CFFF的DSW似乎不支持conda创建虚拟环境（可能一个实例就是一个环境？），因此直接在默认环境下安装依赖

> 安装依赖时需要管理员权限

```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
sudo pip install -e ".[torch,metrics]"
```

配置一下Model Scope下载模型的环境

```sh
export USE_MODELSCOPE_HUB=1
# 更改模型缓存地址
export MODELSCOPE_CACHE=/cpfs01/projects-SSD/cfff-d4f7bbbfa159_SSD/zfy_20301030034/modelscope
pip install modelscope vllm
# 安装vllm时可能导致进程killed，需要降低内存安装
# pip install modelscope vllm --no-cache-dir
```



### 3. 下载模型

在`LLaMa-Factory`目录下新建一个`model_download.py`脚本，内容如下：

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct')
```

命令行中运行该脚本，开始模型下载：

```sh
python model_download.py
```

模型下载好会被存在`/cpfs01/projects-SSD/cfff-d4f7bbbfa159_SSD/zfy_20301030034/models/modelscope`目录下。



### 4. 部署和推理

在`LLaMa-Factory`目录下新建一个`run.sh`脚本，内容如下：

```sh
CUDA_VISIBLE_DEVICES=0 python src/webui.py \
    --model_name_or_path /cpfs01/projects-SSD/cfff-d4f7bbbfa159_SSD/zfy_20301030034/models/modelscope/hub/Qwen/Qwen2___5-7B-Instruct \
    --template qwen \
    --infer_backend vllm \
	--vllm_enforce_eager
# 默认端口为7860
```

即可启动webui界面。利用autodl提供的ssh隧道工具，即可在本地访问云服务器的7860端口。修改下图中的配置，点击加载模型，即可完成部署。

![webui](https://lhcos-84055-1317429791.cos.ap-shanghai.myqcloud.com/博客/LLaMA-Factory/LLaMA-Factory webui.png)

加载完成后，问一下模型他是谁，可以看到他认为自己是阿里云开发的通义千问。

![image-20241216153917516](https://lhcos-84055-1317429791.cos.ap-shanghai.myqcloud.com/博客/LLaMA-Factory/qwen自我认知.png)



## 2. 微调

### 1. 数据集准备

使用[自我认知微调数据集 · 数据集](https://www.modelscope.cn/datasets/swift/self-cognition/)对Qwen-7B-Instruct进行SFT微调。

该自我认知数据集由modelsope swift创建, 可以通过将通配符进行替换：{{NAME}}、{{AUTHOER}}，来创建属于自己大模型的自我认知数据集，总共108条。

![image-20241216151808298](https://lhcos-84055-1317429791.cos.ap-shanghai.myqcloud.com/博客/LLaMA-Factory/modelscope自我认知数据集.png)

不过，该数据集中指令对是以jsonl格式存储的，不符合LLaMA-Factory微调数据集的格式，因此我们首先需要对数据集进行格式的转换。我让cursor给我写了个Python脚本，在进行格式转换的同时顺便进行了通配符的替换。转换后的数据集符合Alpaca格式（详见[LLaMA-Factory/data/README_zh.md ](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md)）

```python
import json

# 定义替换值
REPLACEMENTS = {
    'zh': {
        '{{NAME}}': '小猫',
        '{{AUTHOR}}': '瀚海雪豹'
    },
    'en': {
        '{{NAME}}': 'cattt',
        '{{AUTHOR}}': 'PolarSnowLeopard'
    }
}

def convert_to_alpaca_format(data):
    # 获取语言类型的替换值
    replacements = REPLACEMENTS[data['tag']]
    
    # 替换响应中的占位符
    response = data['response']
    for placeholder, value in replacements.items():
        response = response.replace(placeholder, value)
    
    # 构建 Alpaca 格式的数据
    return {
        'instruction': data['query'],
        'input': '',  # 这个数据集没有额外的输入
        'output': response,
        'system': '',  # 这个数据集没有系统提示词
        'history': []  # 这个数据集没有对话历史
    }

def main():
    # 读取 JSONL 文件
    with open('self_cognition.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 转换每一行数据
    converted_data = []
    for line in lines:
        if line.strip():  # 跳过空行
            data = json.loads(line)
            converted_data.append(convert_to_alpaca_format(data))
    
    # 保存为 JSON 文件
    with open('self_cognition.json', 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

```

完成转换后的`self_cognition.json`中的数据大概长这样：

```json
[
    {
        "instruction": "你是？",
    	"input": "",
    	"output": "我是小猫，由瀚海雪豹训练的人工智能助手。我的目标是为用户提供有用、准确和及时的信息，并通过各种方式帮助用户进	行有效的沟通。请告诉我有什么可以帮助您的呢？",
    	"system": "",
    	"history": []
  	}
    ....
]
```

此外，修改一下`dataset_info.json`中的数据集元信息：

```json
{
    "self_cognition": {
        "file_name": "self_cognition.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
            "history": "history"
        }
    }
}
```

完成修改后，将`self_cognition.json`和`dataset_info.json`两个文件上传到云实例。我放在了`/cpfs01/projects-SSD/cfff-d4f7bbbfa159_SSD/zfy_20301030034/modelscope/hub/datasets/self_cognition`路径下。

### 2. 设置微调参数并进行训练

由于对LLM的训练不是很了解，因此大部分参数我都使用的，默认值。主要是设置一下数据集路径，然后把训练轮数调成了`100`（默认的3没有收敛，可能是因为这个数据集只有108条数据，有点少）

![image-20241216153134999](https://lhcos-84055-1317429791.cos.ap-shanghai.myqcloud.com/博客/LLaMA-Factory/微调设置.png)

![train_loss](https://lhcos-84055-1317429791.cos.ap-shanghai.myqcloud.com/博客/LLaMA-Factory/train_loss.png)

### 3. 推理

训练完成后，切换至`chat`选项卡，选择“检查点路径”后加载模型，就可以和微调之后的模型对话了

![image-20241216153551256](https://lhcos-84055-1317429791.cos.ap-shanghai.myqcloud.com/博客/LLaMA-Factory/微调后推理设置.png)

问一下模型他是谁，可以看到他认为自己是瀚海雪豹开发的小猫而不是阿里云开发的通义千问，这说明我们的微调成功了。

![image-20241216153705014](https://lhcos-84055-1317429791.cos.ap-shanghai.myqcloud.com/博客/LLaMA-Factory/微调后自我认知.png)
