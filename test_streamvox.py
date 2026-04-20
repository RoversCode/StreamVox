from streamvox import TTSEngine

engine = TTSEngine(
    model="s2-pro-4b-gguf",
    license_key="LJJ-CMREFZI-184468df-155e-4417-aab3-33c8c97da6b9",
    device="auto",
)

prompt = engine.make_prompt(
    role_name="demo_role",
    audio_path="example/Condition3.wav",
    prompt_text="所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。",
    persist=False,
)

import soundfile as sf
import numpy as np

chunks = engine.stream(
    "你好，这里是 StreamVox 的快速开始示例。",
    role_name=prompt,
    language="chinese",
    track_performance=True
)

audios = []

for idx, chunk in enumerate(chunks):
    audios.append(chunk)

audios = np.concatenate(audios, axis=-1)
    

sf.write(f"output.wav", audios, engine.runtime.sample_rate)

engine.shutdown()