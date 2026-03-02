# Evaluate model (checkpoint path needed)
python eval_model.py --checkpoint /path/to/checkpoint.pt

# Run IJBC evaluation with ONNX model
python onnx_ijbc.py --model-root qarepvgg_b1_40.onnx --result-dir IJBC_result

# Benchmark ONNX model using TensorRT trtexec
/usr/src/tensorrt/bin/trtexec   --onnx=qarepvgg_b1_40.onnx   --fp16   --memPoolSize=workspace:16384   --iterations=100   --avgTiming=100
