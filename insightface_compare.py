import cv2
import numpy as np
import time
import threading
import queue
from ultralytics import YOLO

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

import os
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
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            if not self.q.full():
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
MODEL_NAME = 'buffalo_m'  # InsightFace 模型名稱
THRESHOLD = 0.35                     

# ==========================================
# 初始化
# ==========================================
def init_insightface():
    print(f"🚀 初始化 InsightFace ({MODEL_NAME})...")
    app = FaceAnalysis(name=MODEL_NAME, providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"🚀 初始化 YOLO11-Nano...")
    yolo_model = YOLO('yolo11n.pt')
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
            
            # 強制使用原圖解析度進行註冊
            h, w = img.shape[:2]
            new_w = int(np.ceil(w / 32) * 32)
            new_h = int(np.ceil(h / 32) * 32)
            app.det_model.input_size = (new_w, new_h)
            
            faces = app.get(img)
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
    return known_embeddings, known_names, known_files

# ==========================================
# 輔助函式
# ==========================================
def draw_recognition(img, box, name, score, color):
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    label = f"{name} ({score:.2f})"
    text_y = box[1] - 10 if box[1] - 10 > 10 else box[1] + 20
    cv2.putText(img, label, (box[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def match_face(face_embedding, known_embeddings, known_names):
    if len(known_embeddings) == 0: return "Unknown", 0.0
    target_embedding = face_embedding / np.linalg.norm(face_embedding)
    known_embeddings_np = np.array(known_embeddings)
    sims = np.dot(known_embeddings_np, target_embedding)
    best_idx = np.argmax(sims)
    max_score = sims[best_idx]
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
    
    img_resized = cv2.resize(img, (640, 640))
    # 2. 調低 YOLO 信心門檻 (conf=0.15)，讓它更敏感
    results = yolo_model(img_resized, classes=[0], conf=0.15, verbose=False)
    
    h_org, w_org = img.shape[:2]
    scale_w = w_org / 640
    scale_h = h_org / 640
    
    DO_RECOGNITION = (frame_counter % 5 == 0)
    
    # 標記現有追蹤對象為未匹配
    for identity in tracked_identities:
        identity['matched_this_frame'] = False

    # 處理 YOLO 偵測結果
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1_real = max(0, int(x1 * scale_w))
        y1_real = max(0, int(y1 * scale_h))
        x2_real = min(w_org, int(x2 * scale_w))
        y2_real = min(h_org, int(y2 * scale_h))
        
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
                h_crop, w_crop = person_crop.shape[:2]
                MAX_INFER_SIZE = 320 
                infer_scale = 1.0
                if w_crop > MAX_INFER_SIZE or h_crop > MAX_INFER_SIZE:
                    infer_scale = MAX_INFER_SIZE / max(w_crop, h_crop)
                    new_w, new_h = int(w_crop * infer_scale), int(h_crop * infer_scale)
                    person_crop_infer = cv2.resize(person_crop, (new_w, new_h))
                else:
                    person_crop_infer = person_crop
                
                h_inf, w_inf = person_crop_infer.shape[:2]
                app.det_model.input_size = (int(np.ceil(w_inf/32)*32), int(np.ceil(h_inf/32)*32))
                faces = app.get(person_crop_infer)
                
                if len(faces) > 0:
                    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    
                    from insightface.utils import face_align
                    norm_crop = face_align.norm_crop(person_crop_infer, face.kps)
                    cv2.imwrite("debug_warped_face_yolo.jpg", norm_crop)

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
    new_w = int(np.ceil(w_org / 32) * 32)
    new_h = int(np.ceil(h_org / 32) * 32)
    app.det_model.input_size = (new_w, new_h)
    
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
def compare_faces(app, yolo_model, target_path, known_embeddings, known_names, known_files): 
    print(f"\n🔍 正在分析目標: {target_path}")
    if not os.path.exists(target_path): return

    ext = os.path.splitext(target_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

    if is_video:
        output_dir = 'result/videos'
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(target_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 計算每幀應該停留的毫秒數 (用於控制播放速度)
        frame_delay = int(1000 / fps)
        
        out_w = width * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = os.path.join(output_dir, f"result_fixed_{os.path.basename(target_path)}")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (out_w, height))
        
        print(f"🎥 開始測試 ({width}x{height} @ {fps:.1f}fps)...")
        
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            frame_yolo = process_yolo_pipeline(app, yolo_model, frame.copy(), known_embeddings, known_names)
            frame_native = process_native_pipeline(app, frame.copy(), known_embeddings, known_names)
            
            combined = np.hstack((frame_yolo, frame_native))
            out.write(combined)
            
            display_scale = 0.5 if width > 1000 else 1.0
            display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow("Left: YOLO+Crop | Right: Native(Full)", display_frame)
            
            # [修正] 智慧等待，控制播放速度
            processing_time = (time.time() - frame_start) * 1000 # ms
            wait_ms = max(1, frame_delay - int(processing_time))
            
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n💾 完成，結果儲存至: {output_filename}")

    else:
        # 圖片模式 (保持不變)
        output_dir = 'result/pictures'
        os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(target_path)
        img_yolo = process_yolo_pipeline(app, yolo_model, img.copy(), known_embeddings, known_names)
        img_native = process_native_pipeline(app, img.copy(), known_embeddings, known_names)
        combined = np.hstack((img_yolo, img_native))
        output_filename = os.path.join(output_dir, f"result_fixed_{os.path.basename(target_path)}")
        cv2.imwrite(output_filename, combined)
        
        h, w = combined.shape[:2]
        display_scale = 0.5 if w > 1000 else 1.0
        cv2.imshow("Result", cv2.resize(combined, (0,0), fx=display_scale, fy=display_scale))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==========================================
# 核心優化：極速測試模式
# ==========================================
def benchmark_performance(app, yolo_model, target_path, known_embeddings, known_names):
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

    fps_yolo = run_pass("YOLO+Crop", process_yolo_pipeline)
    fps_native = run_pass("Native Full", process_native_pipeline)
    
    print(f"\n📊 最終比對結果:")
    print(f"  - YOLO+Crop Pipeline: {fps_yolo:.2f} FPS")
    print(f"  - Native Full Pipeline: {fps_native:.2f} FPS")

if __name__ == "__main__":
    app, yolo_model = init_insightface()
    known_embeddings, known_names, known_files = load_known_faces(app)
    if len(known_embeddings) > 0:
        print("\n請選擇執行模式:")
        print("1. [Benchmark] 極速效能測試 (不顯示畫面)")
        print("2. [Visualize + Benchmark] 視覺化比對後進行效能測試")
        
        choice = input("請輸入選項 (1 或 2): ").strip()
        
        if choice == '1':
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names)
        elif choice == '2':
            compare_faces(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, known_files)
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names)
        else:
            print("❌ 無效選項，結束程式")
    else:
        print("⚠️ 資料庫為空")