# 支持流式推理(非标准openai)
# 支持非流式推理(标准OPENAI)
import io
import time
import torch
from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="CosyVoice[XDF] TTS API")

# 初始化模型 (按实际模型加载方式修改)
cosy_voice = None
device = "cuda" if torch.cuda.is_available() else "cpu"

voice_path='./voices'

class TTSRequest(BaseModel):
    model: str = "tts-1"  # 保持OpenAI兼容的model名称
    voice: str = "alloy"  # 为兼容保留参数，实际使用CosyVoice的默认声音
    input: str
    response_format: Literal["mp3", "flac", "wav"] = "mp3"
    speed: float = 1.0

model_dir = 'pretrained_models/CosyVoice2-0.5B'
   
def get_voices():
    path='./voices'
    wav_files = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.wav')]
    return wav_files

voice_names = get_voices()

print(f"Loading model {model_dir} ...")
cosyvoice = CosyVoice2(
    model_dir=model_dir, 
    load_jit=False, 
    load_trt=False, 
    fp16=False
)

# 加载音色
for voice_name in voice_names:
    print(f'Loading voice: {voice_name}')
    wav_file=f'./voices/{voice_name}.wav'
    prompt_file=f'./voices/{voice_name}.txt'
    prompt_speech_16k = load_wav(wav_file, 16000)
    prompt_text = open(prompt_file).read()
    cosyvoice.add_zero_shot_spk(
        prompt_text=prompt_text,
        prompt_speech_16k=prompt_speech_16k,
        zero_shot_spk_id=voice_name
    )

# print(cosyvoice.list_available_spks())

print(f'@XDF@模型: {model_dir} 已加载')

    
@app.get("/v1/voices")
async def get_voices()->list:
    return voice_names

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    # 参数验证
    if len(request.input) == 0:
        raise HTTPException(400, "Input text cannot be empty")
    if len(request.input) > 4096:
        raise HTTPException(400, "Input text too long (max 4096 characters)")
    if not 0.50 <= request.speed <= 2.0:
        raise HTTPException(400, "Speed must be between 0.50 and 2.0")
    spk_id=request.voice
    if spk_id not in voice_names:
        spk_id = 'default'
    try:
        # 生成语音片段
        audio_segments = []
        for i, segment in enumerate(cosyvoice.inference_zero_shot(
            tts_text=request.input, 
            prompt_text='',
            prompt_speech_16k='',  
            zero_shot_spk_id=spk_id,
            speed=request.speed, 
            stream=False)):
            
            # 收集音频张量
            audio_segments.append(segment['tts_speech'])

        # 合并所有片段（单声道）
        merged_audio = torch.cat(audio_segments, dim=1)  
        merged_audio = merged_audio.numpy().squeeze()   

        # 转换格式并写入内存
        buffer = io.BytesIO()
        sf.write(
            buffer,
            merged_audio,
            cosyvoice.sample_rate, 
            format=request.response_format
        )
        buffer.seek(0)

        # MIME 类型映射
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