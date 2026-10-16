python tools/benchmark_streamvox_models.py \
  --model ckpts/moss-tts-nano-onnx \
  --model-key moss-nano-cpu \
  --display-name "Moss Nano ONNX CPU" \
  --stream-kwargs-json '{"mode":0}' \
  --engine-device cpu
