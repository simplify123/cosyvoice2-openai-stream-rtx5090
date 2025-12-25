# cosyvoice2-openai-stream-rtx5090
##### 应该是首个全面支持5090显卡、并支持openai接口标准、支持双向流式的镜像了，折腾了一个多月...

## 🚀 01 镜像特点

#### 🎯 1、高性能显卡支持
支持 RTX 5090 显卡（Blackwell SM120 架构），为你的计算任务提供强劲动力，轻松应对复杂场景。

#### 🔄 2、实时双向流式
实现数据的实时双向传输，确保信息的即时交互与同步，提升工作效率。在ubuntu原生系统中（注意不要使用wsl或者docker desktop），RTF（实时因子）约为 0.4。

#### 🌐 3、OpenAI 接口标准
完全兼容 OpenAI 接口标准，无缝对接各类 AI 应用(如AIRI数字人、Awesome Digital Human数字人、Super Agent Party数字人等)，拓展无限可能。

## 📦 02 构建镜像

#### 开始构建之前，请先完成几个内容的下载
#### 1、third_party/Matcha-TTS
#### 2、模型下载
```python
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```
#### 3、torch+cuda的whl本地文件下载，放入根目录whl文件夹中
torch-2.8.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl

torchaudio-2.8.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl

torch_tensorrt-2.8.0+cu129-cp310-cp310-manylinux_2_28_x86_64.whl

#### 4、开始构建镜像
```bash
docker build -f Dockerfile-devel.optimized -t cosyvoice2-openai-api-stream-simplify123:latest .
```

## 🎉 03 运行服务
```bash
docker compose up -d
```
docker-compose.yml文件里设置了三个环境变量，可以控制fp16、jit、trt的开启关闭，显存不够的情况下，建议关闭trt
### 💡 服务启动后，通过api进行调用
#### API接口信息配置：
API地址：http://your_ip:51870/v1  
模型ID: tts-1  
API密钥：dummy_key(其实是随便填的)  
音色：jok(可通过音色列表查看需要的角色，添加音色的话，就是往根目录下的voices文件夹里放入音频和对应的文本文档就可以了)

#### 音色列表：
http://your_ip:51870/v1/voices

## 🛠️ 04 已知问题
目前经过测试，在全流式情况下，会有音爆现象，技术有限，不知道如何解决，还有vllm加速也没搞定，据说要在5090显卡上编译vllm源码，暂时没时间折腾。
在按照标点拆分句子，客户端流式情况下，首包延迟1.4~1.6秒，声音效果很好。在dify v1.11.1中(其他版本应该也行，没测试)可以通过Text To Speech插件进行音频文件的生成。

## 📈 运行效果
```bash
==========
== CUDA ==
==========
CUDA Version 12.9.0
Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
Loading model pretrained_models/CosyVoice2-0.5B ...
🚀 尝试启用TensorRT加速 (首次启动会转换模型,需5-10分钟)
✅ 开关配置: fp16=False, JIT=True, TRT=False
/opt/conda/envs/cosyvoice/lib/python3.10/site-packages/lightning/fabric/__init__.py:41: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
/opt/conda/envs/cosyvoice/lib/python3.10/site-packages/diffusers/models/lora.py:393: FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.
  deprecate("LoRACompatibleLinear", "1.0.0", deprecation_message)
2025-12-24 02:09:53,527 INFO input frame rate=25
/opt/conda/envs/cosyvoice/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
2025-12-24 02:09:55.386684287 [W:onnxruntime:, transformer_memcpy.cc:111 ApplyImpl] 8 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2025-12-24 02:09:55.389394606 [W:onnxruntime:, session_state.cc:1316 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2025-12-24 02:09:55.389401709 [W:onnxruntime:, session_state.cc:1318 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
text.cc: festival_Text_init
open voice lang map failed
✅ 模型已加载, fp16=False, JIT=True, TRT=False
🔧 成功应用运行时补丁: encoder.forward 将忽略 context 参数
Loading voice: furina
/opt/conda/envs/cosyvoice/lib/python3.10/site-packages/torchaudio/_backend/utils.py:213: UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec` under the hood. Some parameters like ``normalize``, ``format``, ``buffer_size``, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.
  warnings.warn(
Loading voice: jok
Loading voice: ben
Loading voice: nezha
Loading voice: ad
Loading voice: default
Loading voice: yanglan
Loading voice: jialing
Loading voice: dyy
Loading voice: dehua
Loading voice: alloy
Loading voice: luyu
@XDF@模型: pretrained_models/CosyVoice2-0.5B 已加载
2025-12-24 02:10:49,383 DEBUG Using selector: EpollSelector
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:51870 (Press CTRL+C to quit)
INFO:     100.64.0.3:6089 - "GET /v1/audio HTTP/1.1" 404 Not Found
INFO:     100.64.0.3:6089 - "GET /favicon.ico HTTP/1.1" 404 Not Found
INFO:     100.64.0.3:11988 - "GET /v1/audio/voice HTTP/1.1" 404 Not Found
INFO:     100.64.0.3:11988 - "GET /v1/audio/voices HTTP/1.1" 404 Not Found
INFO:     100.64.0.3:13891 - "GET /v1/voices HTTP/1.1" 200 OK

  0%|          | 0/1 [00:00<?, ?it/s]2025-12-24 04:54:54,550 INFO synthesis text 现在是一段声音测试，经过测试，R T X五零九零显卡可以在ubuntu系统下达到实时R T F输出，整体效果不错，下一步，我们将研究整合cosyvoice三。
/workspace/CosyVoice/cosyvoice/cli/model.py:157: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with self.llm_context, torch.cuda.amp.autocast(self.fp16):
/workspace/CosyVoice/cosyvoice/cli/model.py:337: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(self.fp16), self.trt_context_dict[uuid]:
2025-12-24 04:55:02,148 INFO yield speech len 15.24, rtf 0.49856089231536144

100%|██████████| 1/1 [00:07<00:00,  7.61s/it]
100%|██████████| 1/1 [00:07<00:00,  7.61s/it]
INFO:     100.64.0.3:11862 - "POST /v1/audio/speech HTTP/1.1" 200 OK
```


```python
C:\Users\Administrator\Desktop\tts-test\venv\Scripts\python.exe C:\Users\Administrator\Desktop\tts-test\3streaming_playback_openai_unbuffered.py 
🎵 开始请求TTS流式音频(OpenAI SDK + 无缓冲)...
⚡ 首包延迟: 1.17 秒
📦 首包大小: 8192 字节
📊 WAV格式: 3 (1=PCM, 3=Float), 24000Hz, 1声道, 32bit
🎵 播放块大小: 9600 字节 (约0.1秒)
🔊 预缓冲完成! 已缓冲 0.2 秒音频,开始播放!
✅ 音频流初始化: 24000Hz, 1声道, 16-bit PCM
🔊 # 20 | 已播放: 2.0秒 | 缓冲: 0.0秒 (   4564B) | 接收: 192.0KB
🔊 # 40 | 已播放: 4.0秒 | 缓冲: 0.0秒 (    980B) | 接收: 376.0KB
🔊 # 60 | 已播放: 6.0秒 | 缓冲: 0.1秒 (   5588B) | 接收: 568.0KB
🔊 # 80 | 已播放: 8.0秒 | 缓冲: 0.0秒 (   2004B) | 接收: 752.0KB
🔊 #100 | 已播放: 10.0秒 | 缓冲: 0.1秒 (   6612B) | 接收: 944.0KB
🔊 #120 | 已播放: 12.0秒 | 缓冲: 0.0秒 (   3028B) | 接收: 1128.0KB
🔊 #140 | 已播放: 14.0秒 | 缓冲: 0.1秒 (   7636B) | 接收: 1320.0KB
🔊 #160 | 已播放: 16.0秒 | 缓冲: 0.0秒 (   4052B) | 接收: 1504.0KB
🔊 #180 | 已播放: 18.0秒 | 缓冲: 0.0秒 (    468B) | 接收: 1688.0KB
🔊 #200 | 已播放: 20.0秒 | 缓冲: 0.1秒 (   5076B) | 接收: 1880.0KB
🔊 #220 | 已播放: 22.0秒 | 缓冲: 0.0秒 (   1492B) | 接收: 2064.0KB
🔊 #240 | 已播放: 24.0秒 | 缓冲: 0.1秒 (   6100B) | 接收: 2256.0KB
🔊 #260 | 已播放: 26.0秒 | 缓冲: 0.0秒 (   2516B) | 接收: 2440.0KB
🔊 #280 | 已播放: 28.0秒 | 缓冲: 0.1秒 (   7124B) | 接收: 2632.0KB
🔊 #300 | 已播放: 30.0秒 | 缓冲: 0.0秒 (   3540B) | 接收: 2816.0KB
🔊 #320 | 已播放: 32.0秒 | 缓冲: 0.1秒 (   8148B) | 接收: 3008.0KB
🔊 #340 | 已播放: 34.0秒 | 缓冲: 0.0秒 (   4564B) | 接收: 3192.0KB
🔊 #360 | 已播放: 36.0秒 | 缓冲: 0.0秒 (    980B) | 接收: 3376.0KB
🔊 #380 | 已播放: 38.0秒 | 缓冲: 0.1秒 (   5588B) | 接收: 3568.0KB
🔊 #400 | 已播放: 40.0秒 | 缓冲: 0.0秒 (   2004B) | 接收: 3752.0KB
🔊 最后一块: 960B
⏳ 等待播放完成...

✅ 音频播放完成!

📊 统计信息:
   音频时长: 41.50 秒
   播放块数: 415 个
   数据接收: 3892.5 KB
   总耗时: 43.58 秒
   首包延迟: 1.17 秒

Process finished with exit code 0
```
