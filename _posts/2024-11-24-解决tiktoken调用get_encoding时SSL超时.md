---
title: 解决tiktoken库调用get_encoding时SSL超时
date: 2024-11-24 20:43:12 +0800
categories: [Python, tiktoken]
tags: [Python, tiktoken]     # TAG names should always be lowercase
author: PolarSnowLeopard
---

# 解决tiktoken库调用get_encoding时SSL超时

最近在看*Build a Large Language Model (From Scratch)* 这本书。在该书的第二章中，作者尝试使用`tiktoken`库构建一个tokenizer。然而，当我执行以下代码时，程序报错。

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
```

```shell
ConnectTimeout: HTTPSConnectionPool(host='openaipublic.blob.core.windows.net', port=443): Max retries exceeded with url: /encodings/gpt2.tiktoken (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x7fd41b819630>, 'Connection to openaipublic.blob.core.windows.net timed out. (connect timeout=None)'))
```

这个错误的根本原因是构建tokenizer时`tiktoken`库尝试下载词表文件遇到网络问题而失败。一个可行的解决方案时先手动下载文件到本地，然后让`tiktoken`直接从本地文件读取数据并构建tokenizer。

## 1. 获取词表文件url

> First, let's grab the tokenizer blob URL from the source on your remote machine. If we trace the `get_encoding` function, we find it calls a function from `tiktoken_ext.openai_public` which has the blob URIs for each encoder. Identify the correct function, then print the source

首先需要查看一下构建我们所需的tokenizer需要哪些词表文件。我这里需要使用构建名为`gpt2`的tokenizer。从下面的输出信息可以看到，还有 `o200k_base`, `p50k_base`等可供选择。结果显示，构建`gpt2`tokenizer需要下载`vocab.bpe`和`encoder.json`两个文件。



```python
import tiktoken_ext.openai_public
import inspect

print(dir(tiktoken_ext.openai_public))
# The encoder we want is cl100k_base, we see this as a possible function

print(inspect.getsource(tiktoken_ext.openai_public.gpt2))
# The URL should be in the 'load_tiktoken_bpe function call'
```

运行结果：

```shell
['ENCODING_CONSTRUCTORS', 'ENDOFPROMPT', 'ENDOFTEXT', 'FIM_MIDDLE', 'FIM_PREFIX', 'FIM_SUFFIX', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'cl100k_base', 'data_gym_to_mergeable_bpe_ranks', 'gpt2', 'load_tiktoken_bpe', 'o200k_base', 'p50k_base', 'p50k_edit', 'r50k_base']
def gpt2():
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        encoder_json_file="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        vocab_bpe_hash="1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        encoder_json_hash="196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    )
    return {
        "name": "gpt2",
        "explicit_n_vocab": 50257,
        # The pattern in the original GPT-2 release is:
        # r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # This is equivalent, but executes faster:
        "pat_str": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 50256},
    }
```



## 2. 手动下载词表文件并保存到本地

根据步骤1获得的url，手动下载词表文件并保存到本地。



## 3. 复制并重命名文件

新建一个文件夹`.tiktoken`，将下载的词表文件复制至该文件夹。重命名各文件，新的文件值可以通过执行以下代码获取。`blobpath`是步骤1中获取的该文件对应的url值。

```python
import hashlib

blobpath = "your_blob_url_here"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
print(cache_key)
```

比如对于我刚刚下载的`encoder.json`文件，结果如下：

```python
import hashlib

blobpath = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
cache_key = hashlib.sha1(blobpath.encode()).hexdigest()
print(cache_key)
```

```shell
6c7ea1a7e38e3a7f062df639a5b80947f075ffe6
```

于是将`encoder.json`重命名为`6c7ea1a7e38e3a7f062df639a5b80947f075ffe6`（注意，重命名后的文件不带.json后缀）。



## 4. 环境变量中设置tiktoken cache

执行以下代码，指定tiktoken cache为`.titoken`文件夹。注意，每次使用tiktoken库时都要运行下述代码。

```python
import os

tiktoken_cache_dir = "path_to_folder_containing_tiktoken_file"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# validate
assert os.path.exists(os.path.join(tiktoken_cache_dir, cache_key))
```



## 5. 使用tiktoken库

现在应该可以正常使用tiktoken库构建tokenizer。

```python
encoding = tiktoken.get_encoding("gpt2")
encoding.encode("Hello, world")
```

```shell
[15496, 11, 995]
```



## 参考资料

[[1] SSLError: HTTPSConnectionPool(host='openaipublic.blob.core.windows.net', port=443): Max retries exceeded with url · Issue #281 · openai/tiktoken (github.com)](https://github.com/openai/tiktoken/issues/281)

[[2] python - how to use tiktoken in offline mode computer - Stack Overflow](https://stackoverflow.com/questions/76106366/how-to-use-tiktoken-in-offline-mode-computer)
