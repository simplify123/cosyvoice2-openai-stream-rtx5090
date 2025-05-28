import os
from openai import OpenAI
from datetime import datetime
# 创建输出目录
os.makedirs("C:/temp", exist_ok=True)

base_url='http://devserver:51870/v1'
api_key='key'
voice= 'luyu'
# 初始化客户端
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)
start_time = datetime.now()
# 调用 TTS 接口
response = client.audio.speech.create(
    input='当你悟透了人性、洞悉了交往的规律，会发现这世上，就没有处不好的关系。',
    model='tts-1',
    voice=voice,
    speed=1.0, 
    response_format='mp3'
)

# 检查是否成功
if not response.response.is_success:
    raise Exception(f"请求失败，状态码: {response.response.status_code}")

# 获取二进制内容并写入文件
with open(f"C:/temp/0528_{voice}.mp3", "wb") as f:
    f.write(response.response.content)

ts = datetime.now() - start_time
print(f"耗时: {ts.total_seconds()} 秒")
print(f"音频文件已成功保存至: C:/temp/0528_{voice}.mp3")
