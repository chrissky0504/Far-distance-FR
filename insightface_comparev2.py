import cv2
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO
import os
import onnxruntime as ort
import torch # [新增] 引入 PyTorch 以支援 GPU 操作

if not hasattr(np, 'bool'):
    np.bool = bool
# ---------------------------------------------------------
# 強制使用 FP16 (ONNX Runtime TensorRT/CUDA)
# ---------------------------------------------------------
os.environ.setdefault("ORT_TENSORRT_FP16_ENABLE", "1")
os.environ.setdefault("ORT_CUDA_FP16_ENABLE", "1")

# ---------------------------------------------------------
# 修正 NumPy 2.0 相容性問題
# ---------------------------------------------------------
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void]
    }

import insightface
from insightface.app import FaceAnalysis

# ==========================================
# 優化工具: 多執行緒影片讀取 (解決讀取瓶頸)
# ==========================================
class VideoCaptureThreading:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=4) # 緩衝區
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while self.running:
            if not self.q.full():
                # [修正] 只有在佇列未滿時才讀取，確保不丟失幀數
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                self.q.put(frame)
            else:
                time.sleep(0.005) # 避免 CPU 空轉

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def is_opened(self):
        return self.cap.isOpened()
    
    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# ==========================================
# 設定區域
# ==========================================
KNOWN_FACES_DIR = 'captured_faces'
TARGET_IMAGE_PATH = 'muti1.mp4'  # 可更改為影片或圖片路徑
MODEL_NAME = 'my_arcface_pack'  # InsightFace 模型名稱
THRESHOLD = 0.30                     
DISPLAY_EVERY_N = 1

# ==========================================
# 初始化
# ==========================================
def init_insightface(model_name=MODEL_NAME):
    # 將 Provider 優先順序改為 CUDA 優先，暫時避開 TensorRT 的重建問題
    available = ort.get_available_providers()
    # preferred = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
    trt_options = {
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache'
    }
    preferred = [('TensorrtExecutionProvider', trt_options), 'CUDAExecutionProvider']
    providers = []
    for p in preferred:
        name = p[0] if isinstance(p, tuple) else p
        if name in available:
            providers.append(p)
    if not providers:
        providers = ['CPUExecutionProvider']
    print(f"🚀 初始化 InsightFace ({model_name})...")

    print(f"✅ 使用 Providers: {providers}")
    app = FaceAnalysis(name=model_name, providers=providers)
    
    # 對 TensorRT 進行更詳細的配置
    app.prepare(ctx_id=0, det_size=(640, 640))
    # 關閉 warmup，因為它可能會導致奇怪的鎖死問題，特別是在 TensorRT 剛初始化的時候 
    # 或者我們可以手動 warmup

    # 💡 建立一個單獨的辨識模型引用，專供 YOLO 模式使用
    recognition_model = app.models['recognition']

    print(f"🚀 載入 YOLO11-Nano (建議使用 .engine 檔)...")
    yolo_model = YOLO('yolo11n.pt', task='detect') 
    
    # [優化] 確保 YOLO 模型在 GPU 上運行
    if torch.cuda.is_available():
        yolo_model.to('cuda')
        
    return app, yolo_model

# ==========================================
# 載入資料庫
# ==========================================
def load_known_faces(app):
    known_embeddings = []
    known_names = []
    known_files = []
    
    print(f"📂 正在從 '{KNOWN_FACES_DIR}' 載入人臉資料...")
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"❌ 資料夾 {KNOWN_FACES_DIR} 不存在")
        return [], [], []

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"  👤 處理人物: {name}")
        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            filepath = os.path.join(person_dir, filename)
            img = cv2.imread(filepath)
            if img is None: continue

            # 將註冊照片填充到 640x640，保持長寬比
            h, w = img.shape[:2]
            img_pad = np.zeros((640, 640, 3), dtype=np.uint8)
            scale = min(640 / w, 640 / h)
            nw, nh = int(w * scale), int(h * scale)
            img_rs = cv2.resize(img, (nw, nh))
            img_pad[:nh, :nw, :] = img_rs

            print(f"    ➡️ 正在處理: {filename} (已填充至 640x640 以加速載入)")
            faces = app.get(img_pad)
            print(f"    ✅ 完成處理: {filename}")
            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                embedding = faces[0].embedding
                embedding = embedding / np.linalg.norm(embedding)
                known_embeddings.append(embedding)
                known_names.append(name)
                known_files.append(filename)
            else:
                print(f"    ⚠️ 無法偵測到人臉: {filename}")
                
    print(f"✨ 資料庫載入完成，共 {len(known_embeddings)} 筆特徵")
    
    # [優化] 將特徵庫轉換為 GPU Tensor，此後所有比對運算皆在 GPU 進行 (節省 CPU)
    if known_embeddings:
        # 轉換為 FP16 以加速並節省顯存
        known_embeddings_tensor = torch.tensor(np.array(known_embeddings), dtype=torch.float16)
        if torch.cuda.is_available():
            known_embeddings_tensor = known_embeddings_tensor.cuda()
    else:
        known_embeddings_tensor = None
        
    return known_embeddings_tensor, known_names, known_files

# ==========================================
# 輔助函式
# ==========================================
def draw_recognition(img, box, name, score, color):
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    label = f"{name} ({score:.2f})"
    text_y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 20
    cv2.putText(img, label, (box[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# [優化] 重寫為 GPU 版本比對函數
def match_face(face_embedding, known_embeddings_tensor, known_names):
    if known_embeddings_tensor is None: return "Unknown", 0.0
    
    # 1. 將單一人臉特徵轉為 Tensor 並移至 GPU
    target_tensor = torch.tensor(face_embedding, dtype=torch.float16)
    if torch.cuda.is_available():
        target_tensor = target_tensor.cuda()
    
    # 2. 正規化 (GPU)
    target_tensor = target_tensor / target_tensor.norm()
    
    # 3. 矩陣運算計算相似度 (GPU)
    sims = torch.matmul(known_embeddings_tensor, target_tensor)
    
    # 4. 找出最大值和索引
    best_idx = torch.argmax(sims).item()
    max_score = sims[best_idx].item()
    
    if max_score > THRESHOLD:
        return known_names[best_idx], max_score
    return "Unknown", max_score

# 計算兩個框的中心點距離 (用於簡單追蹤)
def get_center_distance(box1, box2):
    cx1, cy1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    cx2, cy2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    return ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5

# ==========================================
# Pipeline 1: YOLO + Crop + Memory (修復閃現問題)
# ==========================================
# 記憶體結構：列表存放 {'center': (x, y), 'name': str, 'score': float, 'color': tuple, 'miss_count': int, 'box': list}
tracked_identities = [] 
frame_counter = 0

def process_yolo_pipeline(app, yolo_model, img, known_embeddings, known_names):
    global frame_counter, tracked_identities
    frame_counter += 1
    t0 = time.time()
    
    # [優化] 移除很吃 CPU 的 cv2.resize，直接將原始圖片傳給 YOLO
    # 設定 imgsz=640 讓 YOLO 內部處理縮放 (通常比外部 CPU resize 更快)
    # 這樣回傳的 box 座標會自動對應回原圖尺寸，無需再手動換算
    results = yolo_model(img, classes=[0], conf=0.15, verbose=False, imgsz=640)
    
    h_org, w_org = img.shape[:2]
    # [優化] 移除了 scale_w/h 計算，因為 YOLO 直接回傳原圖座標
    
    DO_RECOGNITION = (frame_counter % 5 == 0)
    
    # 標記現有追蹤對象為未匹配
    for identity in tracked_identities:
        identity['matched_this_frame'] = False

    # 處理 YOLO 偵測結果
    for box in results[0].boxes:
        # [優化] 改為直接獲取座標 (無需縮放乘法)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1_real = max(0, x1)
        y1_real = max(0, y1)
        x2_real = min(w_org, x2)
        y2_real = min(h_org, y2)
        
        # 1. 擴大裁切範圍 (Padding) —— 確保頭頂不被切掉
        box_h = y2_real - y1_real
        box_w = x2_real - x1_real
        pad_top = int(box_h * 0.15) 
        pad_bot = int(box_h * 0.05)
        pad_side = int(box_w * 0.10)

        x1_real = max(0, x1_real - pad_side)
        y1_real = max(0, y1_real - pad_top)
        x2_real = min(w_org, x2_real + pad_side)
        y2_real = min(h_org, y2_real + pad_bot)
        
        curr_center = ((x1_real+x2_real)/2, (y1_real+y2_real)/2)
        
        # 尋找匹配的追蹤對象
        matched_identity = None
        min_dist = 100 # 距離門檻
        for identity in tracked_identities:
            dist = ((curr_center[0]-identity['center'][0])**2 + (curr_center[1]-identity['center'][1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                matched_identity = identity
        
        if matched_identity:
            matched_identity.update({
                'center': curr_center,
                'box': [x1_real, y1_real, x2_real, y2_real],
                'matched_this_frame': True,
                'miss_count': 0
            })
        else:
            # 新發現的對象
            new_id = {
                'center': curr_center,
                'box': [x1_real, y1_real, x2_real, y2_real],
                'name': "Unknown",
                'score': 0.0,
                'color': (128, 128, 128),
                'matched_this_frame': True,
                'miss_count': 0
            }
            tracked_identities.append(new_id)
            matched_identity = new_id

        if DO_RECOGNITION:
            # --- 執行辨識 ---
            person_crop = img[y1_real:y2_real, x1_real:x2_real]
            if person_crop.size > 0:
                # 💡 統一輸入尺寸為 112x112 以配合 ArcFace/QARepVGG
                face_img = cv2.resize(person_crop, (112, 112))

                faces = app.get(face_img)
                if len(faces) > 0:
                    face = faces[0]
                    final_name, final_score = match_face(face.embedding, known_embeddings, known_names)
                    matched_identity['name'] = final_name
                    matched_identity['score'] = final_score
                    matched_identity['color'] = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)

    # 3. 慣性 (Inertia): 處理未匹配對象並繪製所有活躍對象
    new_tracked_identities = []
    for identity in tracked_identities:
        if not identity['matched_this_frame']:
            identity['miss_count'] += 1
        
        if identity['miss_count'] < 10: # 容忍 10 幀漏抓 (慣性存活)
            new_tracked_identities.append(identity)
            # 繪製結果 (包含慣性框)
            if identity['name'] != "Unknown" or identity['score'] > 0:
                draw_recognition(img, identity['box'], identity['name'], identity['score'], identity['color'])
            else:
                cv2.rectangle(img, (identity['box'][0], identity['box'][1]), (identity['box'][2], identity['box'][3]), identity['color'], 1)
    
    tracked_identities = new_tracked_identities

    t1 = time.time()
    fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
    status = "Recog" if DO_RECOGNITION else "Track"
    cv2.putText(img, f"YOLO+Crop: {fps:.1f} FPS ({status})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img

# ==========================================
# Pipeline 2: Native InsightFace (修復偵測失敗問題)
# ==========================================
def process_native_pipeline(app, img, known_embeddings, known_names):
    # Native 模式不跳幀，為了顯示真實的慘烈速度
    # 也不縮放，為了顯示真實的偵測能力
    
    t0 = time.time()
    h_org, w_org = img.shape[:2]
    
    # [修正] 移除 MAX_INFER_SIZE 限制，或是設得很大
    # 這樣才能在 Native 模式偵測到 20 公尺遠的人
    
    # 確保是 32 的倍數即可
    # new_w = int(np.ceil(w_org / 32) * 32)
    # new_h = int(np.ceil(h_org / 32) * 32)
    # app.det_model.input_size = (new_w, new_h)
    
    faces = app.get(img)
    if len(faces) > 0:
        face = faces[0]
        from insightface.utils import face_align
        norm_crop = face_align.norm_crop(img, face.kps)
        cv2.imwrite("debug_warped_face_native.jpg", norm_crop)

    for face in faces:
        name, score = match_face(face.embedding, known_embeddings, known_names)
        box = face.bbox.astype(int)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        draw_recognition(img, box, name, score, color)

    t1 = time.time()
    fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
    cv2.putText(img, f"Native Full: {fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img

# ==========================================
# 比對功能 (修復影片速度問題)
# ==========================================
def compare_faces(app, yolo_model, target_path, known_embeddings, known_names, known_files, mode='all'): 
    print(f"\n🔍 正在分析目標: {target_path} (Mode: {mode})")
    if not os.path.exists(target_path): return

    ext = os.path.splitext(target_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    if is_video:
        output_dir = 'result/videos'
        os.makedirs(output_dir, exist_ok=True)
        
        # [恢復] 改回使用標準 cv2.VideoCapture 以確保寫入影片時幀數絕對同步
        # 多執行緒讀取在寫入檔案時若同步控制不當容易導致掉幀(加速)，且 GPU 計算是瓶頸，讀取優化影響有限
        cap = cv2.VideoCapture(target_path)
        
        # 獲取影片尺寸
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 修正：確保 FPS 抓取正確，若無效則預設 30
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        fps = fps_source if fps_source > 0 else 30.0
        print(f"🎬 偵測到影片尺寸: {width}x{height}, FPS: {fps}")

        # [新增] 計算每幀應該停留的毫秒數 (用於控制播放速度，與初版邏輯一致)
        frame_delay = int(1000 / fps)

        output_filename = os.path.join(output_dir, f"result_{mode}_{os.path.basename(target_path)}")
        
        # 根據模式決定輸出寬度
        out_width = width * 2 if mode == 'all' else width

        # 使用 mp4v 編碼器
        out = cv2.VideoWriter(
            output_filename, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (out_width, height)
        )

        frame_idx = 0
        try:
            while True:
                frame_start = time.time() # [新增] 記錄開始時間
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                
                combined = None
                frame_yolo = None
                frame_native = None

                if mode in ['all', 'yolo']:
                    frame_yolo = process_yolo_pipeline(app, yolo_model, frame.copy(), known_embeddings, known_names)
                
                if mode in ['all', 'native']:
                    frame_native = process_native_pipeline(app, frame.copy(), known_embeddings, known_names)
                
                if mode == 'all':
                    combined = np.hstack((frame_yolo, frame_native))
                elif mode == 'yolo':
                    combined = frame_yolo
                elif mode == 'native':
                    combined = frame_native

                out.write(combined)

                if frame_idx % DISPLAY_EVERY_N == 0:
                    h_disp, w_disp = combined.shape[:2]
                    display_scale = 0.5 if w_disp > 1000 else 1.0
                    display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)
                    
                    title = "Result"
                    if mode == 'all': title = "Left: YOLO+Crop | Right: Native(Full)"
                    elif mode == 'yolo': title = "YOLO+Crop Pipeline"
                    elif mode == 'native': title = "Native Full Pipeline"
                    
                    cv2.imshow(title, display_frame)
                    
                    # [修正] 智慧等待，控制播放速度 (移植自初版邏輯)
                    # 計算還需要等待多久才能維持正常 FPS
                    processing_time = (time.time() - frame_start) * 1000 # ms
                    wait_ms = max(1, frame_delay - int(processing_time))

                    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            
        out.release()
        cv2.destroyAllWindows()
        print(f"\n💾 完成，結果儲存至: {output_filename}")

    else:
        # 圖片模式 (保持不變)
        output_dir = 'result/pictures'
        os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(target_path)
        combined = None
        img_yolo = None
        img_native = None

        if mode in ['all', 'yolo']:
            img_yolo = process_yolo_pipeline(app, yolo_model, img.copy(), known_embeddings, known_names)
        
        if mode in ['all', 'native']:
            img_native = process_native_pipeline(app, img.copy(), known_embeddings, known_names)
        
        if mode == 'all':
            combined = np.hstack((img_yolo, img_native))
        elif mode == 'yolo':
            combined = img_yolo
        elif mode == 'native':
            combined = img_native

        output_filename = os.path.join(output_dir, f"result_fixed_{mode}_{os.path.basename(target_path)}")
        cv2.imwrite(output_filename, combined)
        
        h, w = combined.shape[:2]
        display_scale = 0.5 if w > 1000 else 1.0
        cv2.imshow("Result", cv2.resize(combined, (0,0), fx=display_scale, fy=display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==========================================
# 準確率測試功能 (Video Full Frame Mode)
# ==========================================
def evaluate_video_accuracy(models_list, video_path):
    print(f"\n📊 開始多模型影片準確率比對 (Full Frame 模式): {video_path}")
    if not os.path.exists(video_path):
        print(f"❌ 找不到影片檔案: {video_path}")
        return

    all_stats = {}
    for model_name in models_list:
        print(f"\n⚙️ 正在測試模型: {model_name}")
        # 1. 初始化模型
        app, _ = init_insightface(model_name)
        # 2. 根據該模型重新載入資料庫特徵
        known_embeddings, known_names, _ = load_known_faces(app)
        
        cap = cv2.VideoCapture(video_path)
        stats = {"total_faces": 0, "identities": {}, "frame_count": 0}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 使用 Native 模式 (全幀偵測)
            faces = app.get(frame)
            stats["total_faces"] += len(faces)
            stats["frame_count"] += 1
            
            for face in faces:
                name, score = match_face(face.embedding, known_embeddings, known_names)
                stats["identities"][name] = stats["identities"].get(name, 0) + 1
            
            if stats["frame_count"] % 50 == 0:
                print(f"  已處理 {stats['frame_count']} 幀...", end="\r")
        
        cap.release()
        all_stats[model_name] = stats
        print(f"\n✅ 模型 {model_name} 測試完成。")

    if not all_stats: return

    first_model = list(all_stats.keys())[0]
    print("\n📈 最終比對統計 (總幀數: {}):".format(all_stats[first_model]["frame_count"]))
    for model_name, s in all_stats.items():
        print(f"\n🔹 [{model_name}]")
        print(f"  - 總計偵測到人臉次數: {s['total_faces']}")
        sorted_ids = sorted(s["identities"].items(), key=lambda x: x[1], reverse=True)
        print(f"  - 辨識分佈 (TOP 5):")
        for name, count in sorted_ids[:5]:
            print(f"    * {name}: {count} 次")

# ==========================================
# 核心優化：極速測試模式
# ==========================================
def benchmark_performance(app, yolo_model, target_path, known_embeddings, known_names, run_mode='all'):
    global frame_counter, tracked_identities
    
    def run_pass(mode_name, process_func):
        global frame_counter, tracked_identities
        frame_counter = 0
        tracked_identities = []
        
        print(f"\n🔥 開始 {mode_name} 效能測試: {target_path}")
        cap = VideoCaptureThreading(target_path)
        count = 0
        start = time.time()
        
        try:
            while True:
                frame = cap.read()
                if frame is None:
                    if not cap.running: break
                    time.sleep(0.001)
                    continue
                    
                if mode_name == "YOLO+Crop":
                    process_func(app, yolo_model, frame, known_embeddings, known_names)
                else:
                    process_func(app, frame, known_embeddings, known_names)

                count += 1
                if count % 30 == 0:
                    elapsed = time.time() - start
                    print(f"\r🚀 {mode_name} FPS: {count / elapsed:.2f} | 幀數: {count}", end="")
        except KeyboardInterrupt:
            pass
            
        total_time = time.time() - start
        avg_fps = count / total_time if total_time > 0 else 0
        print(f"\n🏁 {mode_name} 平均 FPS: {avg_fps:.2f}")
        cap.release()
        return avg_fps

    fps_yolo = 0
    fps_native = 0

    if run_mode == 'all' or run_mode == 'yolo':
        fps_yolo = run_pass("YOLO+Crop", process_yolo_pipeline)
    
    if run_mode == 'all' or run_mode == 'native':
        fps_native = run_pass("Native Full", process_native_pipeline)
    
    print(f"\n📊 最終比對結果:")
    if run_mode == 'all' or run_mode == 'yolo':
        print(f"  - YOLO+Crop Pipeline: {fps_yolo:.2f} FPS")
    if run_mode == 'all' or run_mode == 'native':
        print(f"  - Native Full Pipeline: {fps_native:.2f} FPS")

if __name__ == "__main__":
    app, yolo_model = init_insightface()
    known_embeddings, known_names, known_files = load_known_faces(app)
    
    # 修正判斷：因為 GPU 優化後 known_embeddings 可能是 None 或 Tensor
    if known_embeddings is not None and len(known_names) > 0:
        print("\n請選擇執行模式:")
        print("1. [All Benchmark] 極速效能測試 (YOLO + Native)")
        print("2. [Visualize + Benchmark] 視覺化比對後進行效能測試 (Dual View)")
        print("3. [Accuracy Comparison] 多模型影片準確率比對 (Full Frame)")
        print("4. [YOLO Visualize + Benchmark] YOLO+Crop 視覺化並測試效能")
        print("5. [Native Visualize + Benchmark] Native Full Frame 視覺化並測試效能")
        
        choice = input("請輸入選項 (1-5): ").strip()
        
        if choice == '1':
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, run_mode='all')
        elif choice == '2':
            compare_faces(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, known_files, mode='all')
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, run_mode='all')
        elif choice == '3':
            # 定義想要比較的模型列表
            models_to_test = ['my_arcface_pack', 'buffalo_m']
            evaluate_video_accuracy(models_to_test, TARGET_IMAGE_PATH)
        elif choice == '4':
            compare_faces(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, known_files, mode='yolo')
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, run_mode='yolo')
        elif choice == '5':
            compare_faces(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, known_files, mode='native')
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, run_mode='native')
        else:
            print("❌ 無效選項，結束程式")
    else:
        print("⚠️ 資料庫為空")
