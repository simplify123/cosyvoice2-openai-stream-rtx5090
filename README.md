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
目前经过测试，cosyvoice2在全流式情况下，会有音爆现象，cosyvoice3在全流式情况下效果很好。cosyvoice3的镜像过几天放出来，正在研究vllm加速推理，据说要在5090显卡上编译vllm源码，暂时没时间折腾。
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

### cosyvoice2服务器+客户端全流式测试（有音爆现象）

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

### cosyvoice3服务器+客户端全流式测试（无音爆现象，效果完美）

```python
Loading voice: yanglan
Loading voice: jialing
Loading voice: dyy
Loading voice: dehua
Loading voice: alloy
Loading voice: luyu
@CosyVoice3@ 模型: pretrained_models/Fun-CosyVoice3-0.5B 已加载
2025-12-25 09:28:22,806 DEBUG Using selector: EpollSelector
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:51870 (Press CTRL+C to quit)
INFO:     100.64.0.3:10534 - "POST /v1/audio/speech?stream=true HTTP/1.1" 200 OK
[Stream] 开始流式生成: Super Agent Party 链接一切！现在流式输出功...

  0%|          | 0/2 [00:00<?, ?it/s]2025-12-25 09:31:33,692 INFO synthesis text Super Agent Party链接一切！现在流式输出功能上已经实现了，符合openai标准，并且已经支持了cosyvoice二和cosyvoice三的api流式接口。尤其是cosyvoice三的流式效果非常好，没有音爆现象，而且首包延迟只有一点二秒左右，完全具备生产能力。
/workspace/CosyVoice/cosyvoice/cli/model.py:101: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
/workspace/CosyVoice/cosyvoice/cli/model.py:406: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(self.fp16):
2025-12-25 09:31:38,404 INFO yield speech len 0.84, rtf 5.60937069711231
[Stream] 首chunk: 80684B
2025-12-25 09:31:38,653 INFO yield speech len 1.0, rtf 0.24660992622375488
2025-12-25 09:31:38,922 INFO yield speech len 1.0, rtf 0.26787495613098145
2025-12-25 09:31:39,381 INFO yield speech len 1.0, rtf 0.4589097499847412
2025-12-25 09:31:39,848 INFO yield speech len 1.0, rtf 0.46615028381347656
2025-12-25 09:31:40,537 INFO yield speech len 1.0, rtf 0.6884362697601318
2025-12-25 09:31:40,783 INFO yield speech len 1.0, rtf 0.2449195384979248
2025-12-25 09:31:41,201 INFO yield speech len 1.0, rtf 0.41713476181030273
2025-12-25 09:31:41,587 INFO yield speech len 1.0, rtf 0.38558077812194824
2025-12-25 09:31:42,157 INFO yield speech len 1.0, rtf 0.5692059993743896
[Stream] 已发送 10 chunk
2025-12-25 09:31:42,593 INFO yield speech len 1.0, rtf 0.43529224395751953
2025-12-25 09:31:43,141 INFO yield speech len 1.0, rtf 0.5472030639648438
2025-12-25 09:31:43,610 INFO yield speech len 1.0, rtf 0.4689295291900635
2025-12-25 09:31:44,237 INFO yield speech len 1.0, rtf 0.6262438297271729
2025-12-25 09:31:44,643 INFO yield speech len 1.0, rtf 0.4047255516052246
2025-12-25 09:31:45,333 INFO yield speech len 1.0, rtf 0.6898794174194336
2025-12-25 09:31:45,874 INFO yield speech len 1.0, rtf 0.5401885509490967
2025-12-25 09:31:46,658 INFO yield speech len 1.0, rtf 0.783289909362793
2025-12-25 09:31:47,139 INFO yield speech len 1.0, rtf 0.48041772842407227
2025-12-25 09:31:48,003 INFO yield speech len 1.0, rtf 0.8633244037628174
[Stream] 已发送 20 chunk
2025-12-25 09:31:48,523 INFO yield speech len 1.0, rtf 0.5194668769836426
2025-12-25 09:31:49,470 INFO yield speech len 1.0, rtf 0.9460289478302002
2025-12-25 09:31:49,918 INFO yield speech len 0.92, rtf 0.4867885423743206

 50%|█████     | 1/2 [00:16<00:16, 16.24s/it]2025-12-25 09:31:49,928 INFO synthesis text 只是Super Agent Party现在角色不能发声，回头还需要使用google的anti gravity里的claude sonnet四点五thinking模型再修一下这个bug。这样数字人就完整啦！
2025-12-25 09:31:50,526 INFO yield speech len 0.84, rtf 0.7117822056724912
2025-12-25 09:31:51,042 INFO yield speech len 1.0, rtf 0.5153264999389648
2025-12-25 09:31:51,456 INFO yield speech len 1.0, rtf 0.41341233253479004
2025-12-25 09:31:51,898 INFO yield speech len 1.0, rtf 0.4415857791900635
2025-12-25 09:31:52,332 INFO yield speech len 1.0, rtf 0.43299078941345215
2025-12-25 09:31:52,804 INFO yield speech len 1.0, rtf 0.47137999534606934
2025-12-25 09:31:53,237 INFO yield speech len 1.0, rtf 0.4324638843536377
[Stream] 已发送 30 chunk
2025-12-25 09:31:53,739 INFO yield speech len 1.0, rtf 0.5014946460723877
2025-12-25 09:31:54,107 INFO yield speech len 1.0, rtf 0.36736011505126953
2025-12-25 09:31:54,663 INFO yield speech len 1.0, rtf 0.5551884174346924
2025-12-25 09:31:55,082 INFO yield speech len 1.0, rtf 0.4181809425354004
2025-12-25 09:31:55,416 INFO yield speech len 0.8, rtf 0.4173198342323303
[Stream] 完成! 共 35 chunk

100%|██████████| 2/2 [00:21<00:00,  9.92s/it]
100%|██████████| 2/2 [00:21<00:00, 10.87s/it]
INFO:     100.64.0.3:10580 - "POST /v1/audio/speech?stream=true HTTP/1.1" 200 OK
[Stream] 开始流式生成: Super Agent Party 链接一切！现在流式输出功...

  0%|          | 0/2 [00:00<?, ?it/s]2025-12-25 09:32:18,873 INFO synthesis text Super Agent Party链接一切！现在流式输出功能上已经实现了，符合openai标准，并且已经支持了cosyvoice二和cosyvoice三的api流式接口。尤其是cosyvoice三的流式效果非常好，没有音爆现象，而且首包延迟只有一点二秒左右，完全具备生产能力。
2025-12-25 09:32:19,471 INFO yield speech len 0.84, rtf 0.7117844763256255
[Stream] 首chunk: 80684B
2025-12-25 09:32:19,887 INFO yield speech len 1.0, rtf 0.4144413471221924
2025-12-25 09:32:20,299 INFO yield speech len 1.0, rtf 0.41211652755737305
2025-12-25 09:32:20,743 INFO yield speech len 1.0, rtf 0.44255590438842773
2025-12-25 09:32:21,178 INFO yield speech len 1.0, rtf 0.4348604679107666
2025-12-25 09:32:21,650 INFO yield speech len 1.0, rtf 0.47144579887390137
2025-12-25 09:32:21,982 INFO yield speech len 1.0, rtf 0.3315269947052002
2025-12-25 09:32:22,485 INFO yield speech len 1.0, rtf 0.5017974376678467
2025-12-25 09:32:22,855 INFO yield speech len 1.0, rtf 0.36909914016723633
2025-12-25 09:32:23,409 INFO yield speech len 1.0, rtf 0.5535023212432861
[Stream] 已发送 10 chunk
2025-12-25 09:32:23,830 INFO yield speech len 1.0, rtf 0.42056918144226074
2025-12-25 09:32:24,362 INFO yield speech len 1.0, rtf 0.5313923358917236
2025-12-25 09:32:24,816 INFO yield speech len 1.0, rtf 0.4534478187561035
2025-12-25 09:32:25,425 INFO yield speech len 1.0, rtf 0.6081831455230713
2025-12-25 09:32:25,812 INFO yield speech len 1.0, rtf 0.3856842517852783
2025-12-25 09:32:26,486 INFO yield speech len 1.0, rtf 0.6742615699768066
2025-12-25 09:32:27,010 INFO yield speech len 1.0, rtf 0.5230085849761963
2025-12-25 09:32:27,775 INFO yield speech len 1.0, rtf 0.7640585899353027
2025-12-25 09:32:28,239 INFO yield speech len 1.0, rtf 0.4631485939025879
2025-12-25 09:32:29,083 INFO yield speech len 1.0, rtf 0.8435287475585938
[Stream] 已发送 20 chunk
2025-12-25 09:32:29,583 INFO yield speech len 1.0, rtf 0.49954867362976074
2025-12-25 09:32:30,505 INFO yield speech len 1.0, rtf 0.9211084842681885
2025-12-25 09:32:30,929 INFO yield speech len 0.52, rtf 0.8147193835331843

 50%|█████     | 1/2 [00:12<00:12, 12.07s/it]2025-12-25 09:32:30,934 INFO synthesis text 只是Super Agent Party现在角色不能发声，回头还需要使用google的anti gravity里的claude sonnet四点五thinking模型再修一下这个bug。这样数字人就完整啦！
2025-12-25 09:32:31,532 INFO yield speech len 0.84, rtf 0.7119161742074149
2025-12-25 09:32:31,948 INFO yield speech len 1.0, rtf 0.41567373275756836
2025-12-25 09:32:32,363 INFO yield speech len 1.0, rtf 0.4139115810394287
2025-12-25 09:32:32,910 INFO yield speech len 1.0, rtf 0.546699047088623
2025-12-25 09:32:33,345 INFO yield speech len 1.0, rtf 0.4335761070251465
2025-12-25 09:32:33,817 INFO yield speech len 1.0, rtf 0.47169971466064453
2025-12-25 09:32:34,148 INFO yield speech len 1.0, rtf 0.33098578453063965
[Stream] 已发送 30 chunk
2025-12-25 09:32:34,752 INFO yield speech len 1.0, rtf 0.6025612354278564





C:\Users\momo\Desktop\tts-test\venv\Scripts\python.exe C:\Users\momo\Desktop\tts-test\4streaming_playback_openai_unbuffered.py 
🎵 开始请求TTS流式音频(OpenAI SDK + 无缓冲)...
⚡ 首包延迟: 1.13 秒
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
🔊 #420 | 已播放: 42.0秒 | 缓冲: 0.1秒 (   6612B) | 接收: 3944.0KB
🔊 最后一块: 2880B
⏳ 等待播放完成...

✅ 音频播放完成!

📊 统计信息:
   音频时长: 42.10 秒
   播放块数: 421 个
   数据接收: 3952.5 KB
   总耗时: 44.18 秒
   首包延迟: 1.13 秒

Process finished with exit code 0
```
