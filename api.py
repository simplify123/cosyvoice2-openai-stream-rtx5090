# cd /d D:\gitcode\cosyvoice-daxiang027
# conda activate cosyvoice-daxiang027
# python api.py
# http://100.64.0.16:51870/v1/voices


# 支持非流式推理(标准OPENAI)
import io
import time
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal
import soundfile as sf
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import os
from soundfile import info as sfinfo
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import asyncio
import uuid
from typing import List, Dict
import os
import math
import aiofiles
import numpy as np
import random
import functools

# ==================== JIT兼容性补丁 ====================
def patch_encoder_forward(flow_model):
    """
    动态修改 flow_model.encoder.forward 实例方法
    使其忽略 context 参数,兼容 JIT 编译后的模型
    """
    if not hasattr(flow_model, 'encoder'):
        print("⚠️  模型没有 encoder 属性,跳过补丁")
        return

    original_forward = flow_model.encoder.forward

    @functools.wraps(original_forward)
    def patched_forward(*args, **kwargs):
        # 移除 JIT 不接受的 context 参数
        kwargs.pop('context', None)
        return original_forward(*args, **kwargs)

    flow_model.encoder.forward = patched_forward
    print("🔧 成功应用运行时补丁: encoder.forward 将忽略 context 参数")

app = FastAPI(title="CosyVoice[XDF] TTS API")

# 初始化模型 (按实际模型加载方式修改)
cosy_voice = None
device = "cuda" if torch.cuda.is_available() else "cpu"

voice_path='./voices'

class TTSRequest(BaseModel):
    model: str = "tts-1"  # 保持OpenAI兼容的model名称
    voice: str = "alloy"  # 为兼容保留参数,实际使用CosyVoice的默认声音
    input: str
    response_format: Literal["mp3", "flac", "wav"] = "mp3"
    speed: float = 1.0
    # stream 参数已移除,改为查询参数

model_dir = 'pretrained_models/CosyVoice2-0.5B'
   
def get_voices():
    path='./voices'
    wav_files = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.wav')]
    return wav_files

voice_names = get_voices()

# 获取环境变量配置 (默认值保持原有逻辑)
load_jit = os.getenv('LOAD_JIT', 'True').lower() == 'true'
load_trt = os.getenv('LOAD_TRT', 'True').lower() == 'true'
fp16 = os.getenv('FP16', 'False').lower() == 'true'

print(f"Loading model {model_dir} ...")
print(f"🚀 尝试启用TensorRT加速 (首次启动会转换模型,需5-10分钟)")
print(f"✅ 开关配置: fp16={fp16}, JIT={load_jit}, TRT={load_trt}")
cosyvoice = CosyVoice2(
    model_dir=model_dir, 
    load_jit=load_jit, 
    load_trt=load_trt,
    fp16=fp16
)
print(f"✅ 模型已加载, fp16={fp16}, JIT={load_jit}, TRT={load_trt}")

# 应用运行时补丁以兼容JIT编译
if hasattr(cosyvoice, 'model') and hasattr(cosyvoice.model, 'flow'):
    patch_encoder_forward(cosyvoice.model.flow)
else:
    print("⚠️  无法找到 cosyvoice.model.flow,跳过补丁")

# 固定随机种子,确保音色一致
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 加载音色
for voice_name in voice_names:
    print(f'Loading voice: {voice_name}')
    wav_file=f'./voices/{voice_name}.wav'
    prompt_file=f'./voices/{voice_name}.txt'
    prompt_speech_16k = load_wav(wav_file, 16000)
    prompt_text = open(prompt_file, encoding='utf-8').read()
    cosyvoice.add_zero_shot_spk(
        prompt_text=prompt_text,
        prompt_speech_16k=prompt_speech_16k,
        zero_shot_spk_id=voice_name
    )

print(f'@XDF@模型: {model_dir} 已加载')

    
@app.get("/v1/voices")
async def get_voices()->list:
    return voice_names


def create_wav_header(sample_rate: int, channels: int, bits_per_sample: int, data_size: int = 0):
    """创建WAV文件头(44字节)"""
    import struct
    
    if data_size == 0:
        data_size = 0xFFFFFFFF - 36
    
    header = bytearray()
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', data_size + 36))
    header.extend(b'WAVE')
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))
    header.extend(struct.pack('<H', 3))  # IEEE float
    header.extend(struct.pack('<H', channels))
    header.extend(struct.pack('<I', sample_rate))
    bytes_per_second = sample_rate * channels * (bits_per_sample // 8)
    header.extend(struct.pack('<I', bytes_per_second))
    block_align = channels * (bits_per_sample // 8)
    header.extend(struct.pack('<H', block_align))
    header.extend(struct.pack('<H', bits_per_sample))
    header.extend(b'data')
    header.extend(struct.pack('<I', data_size))
    return bytes(header)


def generate_audio_stream(request: TTSRequest, spk_id: str, response_format: str):
    """音频流生成器 - 真正的流式输出 (0.1秒延迟)"""
    try:
        # 从0.1秒增大到0.5秒,提高GPU批处理效率,降低RTF
        sample_threshold = cosyvoice.sample_rate * 0.5  # 0.5秒
        
        accumulated_samples = []
        accumulated_length = 0
        is_first_chunk = True
        chunk_count = 0
        
        print(f"[Stream] 开始流式生成: {request.input[:30]}...")
        
        for i, segment in enumerate(cosyvoice.inference_zero_shot(
            tts_text=request.input, 
            prompt_text='',
            prompt_speech_16k='',  
            zero_shot_spk_id=spk_id,
            speed=request.speed, 
            stream=True
        )):
            audio_chunk = segment['tts_speech']
            audio_np = audio_chunk.numpy().squeeze().astype(np.float32)
            
            accumulated_samples.append(audio_np)
            accumulated_length += len(audio_np)
            
            if accumulated_length >= sample_threshold:
                merged_audio = np.concatenate(accumulated_samples)
                
                if response_format == 'wav':
                    if is_first_chunk:
                        header = create_wav_header(
                            sample_rate=cosyvoice.sample_rate,
                            channels=1,
                            bits_per_sample=32,
                            data_size=0
                        )
                        audio_bytes = merged_audio.tobytes()
                        yield header + audio_bytes
                        is_first_chunk = False
                        chunk_count += 1
                        print(f"[Stream] 首chunk: {len(header)+len(audio_bytes)}B")
                    else:
                        audio_bytes = merged_audio.tobytes()
                        yield audio_bytes
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            print(f"[Stream] 已发送 {chunk_count} chunk")
                else:
                    buffer = io.BytesIO()
                    sf.write(buffer, merged_audio, cosyvoice.sample_rate, format=response_format)
                    yield buffer.getvalue()
                    chunk_count += 1
                
                accumulated_samples = []
                accumulated_length = 0
        
        # 发送剩余音频
        if accumulated_samples:
            merged_audio = np.concatenate(accumulated_samples)
            
            if response_format == 'wav':
                if is_first_chunk:
                    header = create_wav_header(
                        sample_rate=cosyvoice.sample_rate,
                        channels=1,
                        bits_per_sample=32,
                        data_size=len(merged_audio) * 4
                    )
                    audio_bytes = merged_audio.tobytes()
                    yield header + audio_bytes
                else:
                    audio_bytes = merged_audio.tobytes()
                    yield audio_bytes
                chunk_count += 1
            else:
                buffer = io.BytesIO()
                sf.write(buffer, merged_audio, cosyvoice.sample_rate, format=response_format)
                yield buffer.getvalue()
        
        print(f"[Stream] 完成! 共 {chunk_count} chunk")
            
    except Exception as e:
        print(f"流式错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Streaming failed: {str(e)}")


@app.post("/v1/audio/speech")
async def generate_speech(
    request: TTSRequest,
    stream: bool = Query(False, description="是否启用流式输出")
):
    # 参数验证
    if len(request.input) == 0:
        raise HTTPException(400, "Input text cannot be empty")
    if len(request.input) > 4096:
        raise HTTPException(400, "Input text too long (max 4096 characters)")
    if not 0.50 <= request.speed <= 2.0:
        raise HTTPException(400, "Speed must be between 0.50 and 2.0")
    
    spk_id = request.voice
    if spk_id not in voice_names:
        spk_id = 'default'
    
    # 流式响应
    if stream:
        if request.response_format not in ['mp3', 'wav']:
            raise HTTPException(400, "Streaming only supports mp3 and wav formats")
        
        mime_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav"
        }
        
        return StreamingResponse(
            generate_audio_stream(request, spk_id, request.response_format),
            media_type=mime_map[request.response_format],
            headers={
                "Content-Disposition": f"attachment; filename={spk_id}.{request.response_format}",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked"
            }
        )
    
    # 非流式响应
    try:
        audio_segments = []
        for i, segment in enumerate(cosyvoice.inference_zero_shot(
            tts_text=request.input, 
            prompt_text='',
            prompt_speech_16k='',  
            zero_shot_spk_id=spk_id,
            speed=request.speed, 
            stream=False)):
            
            audio_segments.append(segment['tts_speech'])

        merged_audio = torch.cat(audio_segments, dim=1)  
        merged_audio = merged_audio.numpy().squeeze()   

        buffer = io.BytesIO()
        sf.write(
            buffer,
            merged_audio,
            cosyvoice.sample_rate, 
            format=request.response_format
        )
        buffer.seek(0)

        mime_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac"
        }
        if request.response_format not in mime_map:
            raise HTTPException(400, f"Unsupported format: {request.response_format}")

        return StreamingResponse(
            content=buffer,
            media_type=mime_map[request.response_format],
            headers={
                "Content-Disposition": f"attachment; filename={spk_id}.{request.response_format}" 
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Audio generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=51870)
