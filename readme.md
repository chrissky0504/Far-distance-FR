# InsightFace & YOLOv11 臉部辨識效能比對工具

本專案旨在比對「原生 InsightFace 全圖偵測」與「YOLOv11 偵測 + 局部裁切辨識」兩種 Pipeline 的效能與準確度。

## 🚀 功能特點
- **雙 Pipeline 比對**：同時執行 YOLO 輔助辨識與原生 InsightFace 辨識。
- **多執行緒讀取**：使用 `VideoCaptureThreading` 解決影片讀取瓶頸，提升 GPU 利用率。
- **極速測試模式**：提供 Benchmark 模式，排除顯示與寫入損耗，測試硬體極限 FPS。
- **慣性追蹤**：在 YOLO 模式下提供簡單的中心點追蹤與漏抓補償。

## 🛠️ 安裝需求
請確保環境中已安裝 CUDA 驅動與相關套件：
```bash
pip install opencv-python numpy ultralytics insightface onnxruntime-gpu
```

## 📂 資料庫準備
將已知人物的照片放入 `captured_faces` 資料夾，結構如下：
```text
captured_faces/
├── Person_A/
│   ├── image1.jpg
│   └── image2.png
└── Person_B/
    └── image1.jpg
```

## 🖥️ 使用方法
1. 修改 `insightface_compare.py` 中的 `TARGET_IMAGE_PATH` 指向您的測試影片或圖片。
2. 執行程式：
   ```bash
   python insightface_compare.py
   ```
3. 根據提示選擇模式：
   - `1`: 執行極速效能測試 (Benchmark)。
   - `2`: 先進行視覺化比對，結束後再執行效能測試。

## 📊 Pipeline 說明
1. **YOLO+Crop (左側)**：
   - 使用 YOLOv11n 偵測人物。
   - 裁切人物區域並縮放後交由 InsightFace 辨識。
   - 每 5 幀執行一次辨識，其餘幀使用慣性追蹤。
2. **Native Full (右側)**：
   - 直接將原始影像交由 InsightFace 進行全圖偵測與辨識。
   - 適合偵測遠距離微小人臉，但運算負擔較重。

## 🔗 相關資源
- **模型權重 / 測試素材**：[Google Drive 下載連結](https://drive.google.com/file/d/1rxf9VH-5fO69LS0IjhIjQqWv7UTS7bxb/view?usp=sharing)
