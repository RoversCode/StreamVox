from streamvox import TTSEngine

engine = TTSEngine(
    model="qwen3-tts-clone-1.7b-gguf",
    license_key="",
    device="auto",
)

prompt = engine.make_prompt(
    role_name="demo_role",
    audio_path="example/Condition3.wav",
    prompt_text="所以今天我想要讨论另外两个问题，很相似。可是却不一样的问题，我相信自己是谁。",
    persist=False,
)

chunks = engine.stream(
    "你好，这里是 StreamVox 的快速开始示例。",
    role_name=prompt,
    language="chinese",
    track_performance=True
)

# import soundfile as sf

for idx, chunk in enumerate(chunks):
    print(chunk)
    # sf.write(f"output_{idx}.wav", chunk, engine.runtime.sample_rate)

engine.shutdown()
