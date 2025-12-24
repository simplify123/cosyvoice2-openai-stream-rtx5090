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
音色：jok(可通过音色列表查看需要的角色，添加音色的话，就是往根目录下的voice文件夹里放入音频和对应的文本文档就可以了)

#### 音色列表：http://your_ip:51870/v1/voices

## 🛠️ 04 已知问题
目前经过测试，在全流式情况下，会有音爆现象，技术有限，不知道如何解决，还有vllm加速也没搞定，据说要在5090显卡上编译vllm源码，暂时没时间折腾。
在按照标点拆分句子，客户端流式情况下，首包延迟1.4~1.6秒，声音效果很好。在dify v1.11.1中(其他版本应该也行，没测试)可以通过Text To Speech插件进行音频文件的生成。
