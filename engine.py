from ultralytics import YOLO
import tensorrt as trt

print(f"目前 TensorRT 版本: {trt.__version__}")

# 確保載入的是原始 pt 檔
model = YOLO("yolo11n.pt")

# 執行匯出，強制指定使用目前環境的 TensorRT
model.export(format="engine", dynamic=False, half=True)