import cv2
import numpy as np
import time
import os
import glob

# Patch np.bool for compatibility with older libraries (like tensorrt)
if not hasattr(np, 'bool'):
    np.bool = np.bool_

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

import insightface
from insightface.app import FaceAnalysis

# ==========================================
# 優化工具: 多執行緒影片讀取 (解決讀取瓶頸)
# ==========================================
gui_warning_printed = False

def safe_imshow(title, img):
    global gui_warning_printed
    try:
        cv2.imshow(title, img)
        return True
    except cv2.error as e:
        if "The function is not implemented" in str(e) or "xcb_window" in str(e):
            if not gui_warning_printed:
                print("\n⚠️ [警告] 目前環境不支援 OpenCV GUI 顯示 (如 Headless/WSL)。將略過視窗顯示，改為純背景處理並儲存檔案。")
                gui_warning_printed = True
        return False

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
TARGET_IMAGE_PATH = 'outdoor2.mp4'  # 可更改為影片或圖片路徑
MODEL_NAME = 'buffalo_m'  # InsightFace 模型名稱
THRESHOLD = 0.35                     

# ==========================================
# 初始化
# ==========================================
def init_insightface(model_name=MODEL_NAME, load_yolo=True):
    print(f"🚀 初始化 InsightFace ({model_name})...")
    
    # 💡 設定 TensorRT 參數，強制開啟 FP16 與引擎快取
    # 確保快取目錄存在
    if not os.path.exists('./trt_cache'):
        os.makedirs('./trt_cache')

    trt_options = {
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,             # 開啟快取，避免每次重開機都要重新編譯
        'trt_engine_cache_path': './trt_cache',      # 引擎存放資料夾
        'trt_max_workspace_size': 16106127360,       # 允許使用最多 15GB Workspace (解決之前記憶體不足的問題)
    }
    
    # 優先使用 TensorRT，若失敗則降級到 CUDA
    providers = [('TensorrtExecutionProvider', trt_options), 'CUDAExecutionProvider']
    
    app = FaceAnalysis(name=model_name, providers=providers)
    
    # 這裡的 det_size 決定了偵測器的解析度
    app.prepare(ctx_id=0, det_size=(1280, 1280)) 
    
    yolo_model = None
    if load_yolo:
        print(f"🚀 初始化 YOLO11-Nano (TensorRT Engine)...")
        # 只要確認這裡檔名正確即可，其他邏輯完全通用
        yolo_model = YOLO('yolo11n.engine')
    
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
            
            # 💡 核心修正：配合 TRT 引擎尺寸，並防止人臉被過度放大
            # 1. 建立一個 1280x1280 的黑底畫布 (符合 app.prepare 設定)
            canvas = np.zeros((1280, 1280, 3), dtype=np.uint8)
            
            # 2. 將註冊照片縮放，但限制最大不能超過 640 
            # (避免臉太大超出 RetinaFace 的 Anchor Box 極限)
            h, w = img.shape[:2]
            scale = min(640 / w, 640 / h)
            nw, nh = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (nw, nh))
            
            # 3. 將照片貼到畫布左上角
            canvas[:nh, :nw, :] = img_resized

            # 4. 使用標準畫布進行偵測，確保 TensorRT 穩定運作
            from insightface.app.common import Face
            from insightface.utils import face_align
            
            bboxes, kpss = app.det_model.detect(canvas, max_num=0)
            faces = []
            if bboxes is not None:
                rec_model = app.models.get('recognition')
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    kps = kpss[i] if kpss is not None else None
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)
                    
                    if rec_model is not None:
                        # 切出對齊的人臉 (112x112)
                        aimg = face_align.norm_crop(canvas, landmark=face.kps, image_size=rec_model.input_size[0])
                        
                        # 💡 核心修改：同時提取「原始版」與「CLAHE版」的特徵
                        # 1. 原始特徵
                        embedding_normal = rec_model.get_feat(aimg).flatten()
                        embedding_normal = embedding_normal / np.linalg.norm(embedding_normal)
                        
                        # 2. CLAHE 特徵 (強制對註冊照片做 CLAHE，無論它原本亮不亮)
                        aimg_clahe_forced = apply_clahe(aimg)
                        embedding_clahe = rec_model.get_feat(aimg_clahe_forced).flatten()
                        embedding_clahe = embedding_clahe / np.linalg.norm(embedding_clahe)
                        
                        # 把兩種特徵都存進資料庫，名字都叫同一個人
                        known_embeddings.append(embedding_normal)
                        known_names.append(name)
                        known_files.append(filename + "_normal")
                        
                        known_embeddings.append(embedding_clahe)
                        known_names.append(name)
                        known_files.append(filename + "_clahe")
            else:
                print(f"    ⚠️ 無法偵測到人臉: {filename}")
                
    print(f"✨ 資料庫載入完成，共 {len(known_embeddings)} 筆特徵")
    return known_embeddings, known_names, known_files

# ==========================================
# 輔助函式
# ==========================================
def smart_clahe(img, pipeline='native'):
    """
    智慧判斷是否需要進行 CLAHE 處理。
    只在「整體太暗」或「對比度極端 (陰陽臉)」時才觸發。
    pipeline: 'yolo' 或 'native'，用於分別計數與儲存對比圖。
    """
    global clahe_count_yolo, clahe_count_native

    # 1. 轉成單通道灰階來計算亮度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 計算平均亮度 (0~255)
    mean_brightness = np.mean(gray)
    
    # 3. 計算亮度標準差 (判斷是否為光影落差極大的陰陽臉)
    std_brightness = np.std(gray)
    
    # 💡 判斷邏輯 (門檻數值可依你的戶外實際情況微調)
    # 如果平均亮度小於 50 (太暗)，或者標準差大於 85 (光影極端不均)
    if mean_brightness < 50 or std_brightness > 85:
        enhanced = apply_clahe(img)

        # 更新對應 Pipeline 的計數器
        if pipeline == 'yolo':
            clahe_count_yolo += 1
            count = clahe_count_yolo
        else:
            clahe_count_native += 1
            count = clahe_count_native

        # 儲存對比圖 (限制最多 CLAHE_SAVE_MAX 張，避免磁碟爆滿)
        if count <= CLAHE_SAVE_MAX:
            save_dir = 'result/clahe'
            os.makedirs(save_dir, exist_ok=True)
            comparison = np.hstack((img, enhanced))
            save_path = os.path.join(
                save_dir,
                f"{pipeline}_{count:04d}_mean{mean_brightness:.0f}_std{std_brightness:.0f}.jpg"
            )
            cv2.imwrite(save_path, comparison)

        return enhanced
    else:
        # 光線正常，直接放行原圖，保護 ArcFace 特徵
        return img
    
def apply_clahe(img):
    """
    對影像應用 CLAHE (Contrast Limited Adaptive Histogram Equalization)
    增強局部對比度，特別適用於人臉光源不均勻的情況。
    """
    if len(img.shape) == 3:
        # 將 BGR 轉換為 LAB 色彩空間
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 建立 CLAHE 物件並應用於 L 通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 合併通道並轉換回 BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # 單通道灰階圖
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

def crop_and_pad_center(img, target_w=1280, target_h=1280):
    """
    將影像裁切並填充至目標大小 (預設 1280x1280)。
    若尺寸大於目標則取中間裁切；若小於目標則用白邊補齊。
    """
    h, w = img.shape[:2]
    
    if w > target_w:
        start_x = (w - target_w) // 2
        img = img[:, start_x:start_x+target_w]
    elif w < target_w:
        pad_w = target_w - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
    h = img.shape[0]
    if h > target_h:
        start_y = (h - target_h) // 2
        img = img[start_y:start_y+target_h, :]
    elif h < target_h:
        pad_h = target_h - h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
    return img

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
clahe_count_yolo = 0
clahe_count_native = 0
CLAHE_SAVE_MAX = 50  # 每個 Pipeline 最多儲存幾張對比圖

def process_yolo_pipeline(app, yolo_model, img, known_embeddings, known_names, draw=True):
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
                # 修正：強制使用 1280x1280 畫布以符合 TensorRT 引擎要求
                target_size = (1280, 1280)
                canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                
                h_crop, w_crop = person_crop.shape[:2]
                
                # 為了保留細節，這裡我們不一定要將小圖放大到 1280，
                # 但必須確保它被放在 1280 的畫布中。
                # 為了避免小臉被過度放大造成失真 (RetinaFace 錨框問題)，我們限制最大縮放
                # 但如果是很小的臉，我們可以適度放大增加特徵
                
                # 計算縮放比例，讓裁切圖適應畫布，但可以設定上限
                scale = min(target_size[0] / w_crop, target_size[1] / h_crop)
                
                # 限制最大放大倍率，避免將極小的臉放大到全螢幕導致誤判
                # 但也確保不會因為太小而被忽略。這裡我們讓它盡量填滿畫布，
                # 因為這是針對 "單一人臉裁切" 的特徵提取
                
                nw, nh = int(w_crop * scale), int(h_crop * scale)
                person_crop_resized = cv2.resize(person_crop, (nw, nh))
                
                canvas[:nh, :nw, :] = person_crop_resized
                
                from insightface.app.common import Face
                from insightface.utils import face_align
                
                bboxes, kpss = app.det_model.detect(canvas, max_num=0)
                faces = []
                if bboxes is not None:
                    rec_model = app.models.get('recognition')
                    for i in range(bboxes.shape[0]):
                        bbox = bboxes[i, 0:4]
                        det_score = bboxes[i, 4]
                        kps = kpss[i] if kpss is not None else None
                        face = Face(bbox=bbox, kps=kps, det_score=det_score)
                        
                        if rec_model is not None:
                            aimg = face_align.norm_crop(canvas, landmark=face.kps, image_size=rec_model.input_size[0])
                            face.embedding = rec_model.get_feat(aimg).flatten()
                        
                        faces.append(face)
                
                if len(faces) > 0:
                    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    
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
            if draw:
                if identity['name'] != "Unknown" or identity['score'] > 0:
                    draw_recognition(img, identity['box'], identity['name'], identity['score'], identity['color'])
                else:
                    cv2.rectangle(img, (identity['box'][0], identity['box'][1]), (identity['box'][2], identity['box'][3]), identity['color'], 1)
    
    tracked_identities = new_tracked_identities

    t1 = time.time()
    fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
    
    if draw:
        status = "Recog" if DO_RECOGNITION else "Track"
        cv2.putText(img, f"YOLO+Crop: {fps:.1f} FPS ({status})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return img

# ==========================================
# Pipeline 2: Native InsightFace (修復偵測失敗問題)
# ==========================================
def process_native_pipeline(app, img, known_embeddings, known_names, draw=True):
    # Native 模式不跳幀，為了顯示真實的慘烈速度
    # 也不縮放，為了顯示真實的偵測能力
    
    t0 = time.time()
    h_org, w_org = img.shape[:2]
    
    # [修正] TensorRT 必須使用固定的 1280x1280 輸入
    # 我們將影像縮放並貼到 1280x1280 的畫布上
    target_size = (1280, 1280)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    scale = min(target_size[0] / w_org, target_size[1] / h_org)
    nw, nh = int(w_org * scale), int(h_org * scale)
    img_resized = cv2.resize(img, (nw, nh))
    
    canvas[:nh, :nw, :] = img_resized
    
    # 此時不需要再設定 app.det_model.input_size，因為它已經是固定的
    # 原本是 faces = app.get(canvas)，現改為先偵測、對齊，再做 CLAHE，最後才做辨識
    from insightface.app.common import Face
    from insightface.utils import face_align
    
    bboxes, kpss = app.det_model.detect(canvas, max_num=0)
    faces = []
    if bboxes is not None:
        rec_model = app.models.get('recognition')
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            
            if rec_model is not None:
                # 對齊切出人臉
                aimg = face_align.norm_crop(canvas, landmark=face.kps, image_size=rec_model.input_size[0])
                # 送入 ArcFace 取特徵
                face.embedding = rec_model.get_feat(aimg).flatten()
            
            faces.append(face)
    
    if len(faces) > 0:
        face = faces[0]
        # 注意：這裡的 kps (關鍵點) 是在 canvas 座標系上的，如果要裁切原圖需要映射回去
        # 為了簡單除錯，這裡暫時略過寫入 debug 圖片的步驟，或將其註解掉
        # from insightface.utils import face_align
        # norm_crop = face_align.norm_crop(img, face.kps)
        # cv2.imwrite("debug_warped_face_native.jpg", norm_crop)

    for face in faces:
        name, score = match_face(face.embedding, known_embeddings, known_names)
        
        # 將 bbox 映射回原圖座標
        box = face.bbox.astype(float)
        box = box / scale
        box = box.astype(int)
        
        if draw:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            draw_recognition(img, box, name, score, color)

    t1 = time.time()
    fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
    
    if draw:
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
        
        cap = cv2.VideoCapture(target_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 計算每幀應該停留的毫秒數 (用於控制播放速度)
        frame_delay = int(1000 / fps)
        
        # 影像將被裁切與填充為 1280x1280
        target_w, target_h = 1280, 1280
        out_w = target_w * 2 if mode == 'all' else target_w
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = os.path.join(output_dir, f"result_{mode}_{os.path.basename(target_path)}")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (out_w, target_h))
        
        print(f"🎥 開始測試 ({target_w}x{target_h} @ {fps:.1f}fps)...")

        total_start = time.time()
        frame_count = 0

        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            frame = crop_and_pad_center(frame)
            
            frame_yolo = None
            frame_native = None
            combined = None

            if mode in ['all', 'yolo']:
                frame_yolo = process_yolo_pipeline(app, yolo_model, frame.copy(), known_embeddings, known_names, draw=True)
            
            if mode in ['all', 'native']:
                frame_native = process_native_pipeline(app, frame.copy(), known_embeddings, known_names, draw=True)
            
            if mode == 'all':
                combined = np.hstack((frame_yolo, frame_native))
            elif mode == 'yolo':
                combined = frame_yolo
            elif mode == 'native':
                combined = frame_native
                
            out.write(combined)
            frame_count += 1
            
            h_disp, w_disp = combined.shape[:2]
            display_scale = 0.5 if w_disp > 1000 else 1.0
            display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)
            
            title = "Result"
            if mode == 'all': title = "Left: YOLO+Crop | Right: Native(Full)"
            elif mode == 'yolo': title = "YOLO+Crop Pipeline"
            elif mode == 'native': title = "Native Full Pipeline"

            shown = safe_imshow(title, display_frame)
            
            # [修正] 智慧等待，控制播放速度 (僅在 GUI 成功啟動時才等待)
            if shown:
                processing_time = (time.time() - frame_start) * 1000 # ms
                wait_ms = max(1, frame_delay - int(processing_time))
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break
        
        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except: pass
        total_time = time.time() - total_start
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 平均處理速度: {avg_fps:.2f} FPS (共 {frame_count} 幀，耗時 {total_time:.1f} 秒)")
        print(f"💾 完成，結果儲存至: {output_filename}")

    else:
        # 圖片模式
        output_dir = 'result/pictures'
        os.makedirs(output_dir, exist_ok=True)
        
        img = cv2.imread(target_path)
        img = crop_and_pad_center(img)
        combined = None
        
        if mode in ['all', 'yolo']:
            img_yolo = process_yolo_pipeline(app, yolo_model, img.copy(), known_embeddings, known_names, draw=True)
            combined = img_yolo
            
        if mode in ['all', 'native']:
            img_native = process_native_pipeline(app, img.copy(), known_embeddings, known_names, draw=True)
            combined = img_native

        if mode == 'all':
            combined = np.hstack((img_yolo, img_native))

        output_filename = os.path.join(output_dir, f"result_{mode}_{os.path.basename(target_path)}")
        cv2.imwrite(output_filename, combined)
        
        h, w = combined.shape[:2]
        display_scale = 0.5 if w > 1000 else 1.0
        
        shown = safe_imshow("Result", cv2.resize(combined, (0,0), fx=display_scale, fy=display_scale))
        if shown:
            cv2.waitKey(0)
            try:
                cv2.destroyAllWindows()
            except: pass
        else:
            print(f"\n💾 圖片處理完成，結果儲存至: {output_filename}")

# ==========================================
# 視覺化模型比較功能 (Native Mode)
# ==========================================
def visual_compare_models(base_model_name, new_model_name, target_path):
    print(f"\n⚔️ 開始模型視覺化對決: {base_model_name} vs {new_model_name}")
    
    # 初始化兩個模型
    print(f"🔹 載入模型 A: {base_model_name}")
    app_a, _ = init_insightface(base_model_name)
    emb_a, names_a, _ = load_known_faces(app_a)
    
    print(f"🔸 載入模型 B: {new_model_name}")
    app_b, _ = init_insightface(new_model_name)
    emb_b, names_b, _ = load_known_faces(app_b)
    
    if not os.path.exists(target_path): return

    output_dir = 'result/videos'
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps)

    #設定輸出影片
    target_w, target_h = 1280, 1280
    output_filename = os.path.join(output_dir, f"compare_{base_model_name}_vs_{new_model_name}_{os.path.basename(target_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (target_w * 2, target_h))
    
    print("🎥 按 'q' 退出...")
    
    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        frame = crop_and_pad_center(frame)
        
        frame_a = process_native_pipeline(app_a, frame.copy(), emb_a, names_a)
        cv2.putText(frame_a, f"Model A: {base_model_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        frame_b = process_native_pipeline(app_b, frame.copy(), emb_b, names_b)
        cv2.putText(frame_b, f"Model B: {new_model_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        combined = np.hstack((frame_a, frame_b))
        out.write(combined)
        
        h_disp, w_disp = combined.shape[:2]
        display_scale = 0.5 if w_disp > 1000 else 1.0
        display_frame = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)
        
        shown = safe_imshow(f"VS: {base_model_name} | {new_model_name}", display_frame)
        
        if shown:
            processing_time = (time.time() - frame_start) * 1000
            wait_ms = max(1, frame_delay - int(processing_time))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
            
    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except: pass
    print(f"\n💾 完成，結果儲存至: {output_filename}")

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
        # 1. 初始化模型 (功能3不需要YOLO，跳過載入以節省時間)
        app, _ = init_insightface(model_name, load_yolo=False)
        # 2. 根據該模型重新載入資料庫特徵
        known_embeddings, known_names, _ = load_known_faces(app)
        
        cap = cv2.VideoCapture(video_path)
        stats = {"total_faces": 0, "identities": {}, "frame_count": 0, "fps": 0.0}
        
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = crop_and_pad_center(frame)
            
            # 使用 Native 模式 (全幀偵測)
            # 必須包含縮放與畫布邏輯以符合 TensorRT 引擎
            h_org, w_org = frame.shape[:2]
            target_size = (1280, 1280)
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            
            scale = min(target_size[0] / w_org, target_size[1] / h_org)
            nw, nh = int(w_org * scale), int(h_org * scale)
            img_resized = cv2.resize(frame, (nw, nh))
            canvas[:nh, :nw, :] = img_resized
            
            # 使用我們手動切解的流程：SCRFD -> Face Crop -> CLAHE -> ArcFace
            from insightface.app.common import Face
            from insightface.utils import face_align
            
            bboxes, kpss = app.det_model.detect(canvas, max_num=0)
            faces = []
            if bboxes is not None:
                rec_model = app.models.get('recognition')
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    kps = kpss[i] if kpss is not None else None
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)
                    
                    if rec_model is not None:
                        aimg = face_align.norm_crop(canvas, landmark=face.kps, image_size=rec_model.input_size[0])
                        face.embedding = rec_model.get_feat(aimg).flatten()
                    
                    faces.append(face)
            
            stats["total_faces"] += len(faces)
            stats["frame_count"] += 1
            
            for face in faces:
                name, score = match_face(face.embedding, known_embeddings, known_names)
                stats["identities"][name] = stats["identities"].get(name, 0) + 1
                
                # [新增] 模擬實際應用中的繪圖開銷，讓 FPS 測試更公平
                # 將 bbox 映射回原圖座標
                box = face.bbox.astype(float)
                box = box / scale
                box = box.astype(int)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                draw_recognition(frame, box, name, score, color)
            
            if stats["frame_count"] % 50 == 0:
                elapsed = time.time() - start_time
                curr_fps = stats["frame_count"] / elapsed if elapsed > 0 else 0
                print(f"  已處理 {stats['frame_count']} 幀... (FPS: {curr_fps:.1f})", end="\r")
        
        total_time = time.time() - start_time
        stats["fps"] = stats["frame_count"] / total_time if total_time > 0 else 0
        
        cap.release()
        all_stats[model_name] = stats
        print(f"\n✅ 模型 {model_name} 測試完成。平均 FPS: {stats['fps']:.2f}")

    if not all_stats: return

    first_model = list(all_stats.keys())[0]
    print("\n📈 最終比對統計 (總幀數: {}):".format(all_stats[first_model]["frame_count"]))
    for model_name, s in all_stats.items():
        print(f"\n🔹 [{model_name}]")
        print(f"  - 效能表現: {s['fps']:.2f} FPS")
        print(f"  - 總計偵測到人臉次數: {s['total_faces']}")
        sorted_ids = sorted(s["identities"].items(), key=lambda x: x[1], reverse=True)
        print(f"  - 辨識分佈 (TOP 5):")
        for name, count in sorted_ids[:5]:
            print(f"    * {name}: {count} 次")

# ==========================================
# 功能9：CLAHE 效果比對 (buffalo_m + Native Mode)
# ==========================================
def evaluate_clahe_impact(video_path, model_name='buffalo_m'):
    """
    比較 CLAHE 套用前後的辨識效果。
    以 buffalo_m + Native 模式為基底，像功能3一樣統計各身份的辨識次數。
    - No CLAHE: 對齊後的人臉直接送入 ArcFace，不做任何強化
    - With CLAHE: 對齊後的人臉套用 smart_clahe (條件式 CLAHE)，再送入 ArcFace
    """
    print(f"\n🔬 開始 CLAHE 效果比對 (模型: {model_name}, Mode: Native)")
    if not os.path.exists(video_path):
        print(f"❌ 找不到影片檔案: {video_path}")
        return

    print(f"🔹 載入模型: {model_name}")
    app, _ = init_insightface(model_name, load_yolo=False)
    known_embeddings, known_names, _ = load_known_faces(app)

    from insightface.app.common import Face
    from insightface.utils import face_align

    all_stats = {}

    for use_clahe in [False, True]:
        label = "With CLAHE" if use_clahe else "No CLAHE"
        print(f"\n⚙️ 正在測試: [{label}]")

        cap = cv2.VideoCapture(video_path)
        stats = {"total_faces": 0, "identities": {}, "frame_count": 0, "fps": 0.0}
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_and_pad_center(frame)

            h_org, w_org = frame.shape[:2]
            target_size = (1280, 1280)
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

            scale = min(target_size[0] / w_org, target_size[1] / h_org)
            nw, nh = int(w_org * scale), int(h_org * scale)
            img_resized = cv2.resize(frame, (nw, nh))
            canvas[:nh, :nw, :] = img_resized

            bboxes, kpss = app.det_model.detect(canvas, max_num=0)
            faces = []
            if bboxes is not None:
                rec_model = app.models.get('recognition')
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, 0:4]
                    det_score = bboxes[i, 4]
                    kps = kpss[i] if kpss is not None else None
                    face = Face(bbox=bbox, kps=kps, det_score=det_score)

                    if rec_model is not None:
                        aimg = face_align.norm_crop(canvas, landmark=face.kps, image_size=rec_model.input_size[0])
                        if use_clahe:
                            aimg = smart_clahe(aimg, pipeline='native')
                        face.embedding = rec_model.get_feat(aimg).flatten()

                    faces.append(face)

            stats["total_faces"] += len(faces)
            stats["frame_count"] += 1

            for face in faces:
                name, score = match_face(face.embedding, known_embeddings, known_names)
                stats["identities"][name] = stats["identities"].get(name, 0) + 1

            if stats["frame_count"] % 50 == 0:
                elapsed = time.time() - start_time
                curr_fps = stats["frame_count"] / elapsed if elapsed > 0 else 0
                print(f"  已處理 {stats['frame_count']} 幀... (FPS: {curr_fps:.1f})", end="\r")

        total_time = time.time() - start_time
        stats["fps"] = stats["frame_count"] / total_time if total_time > 0 else 0
        cap.release()
        all_stats[label] = stats
        print(f"\n✅ [{label}] 測試完成。平均 FPS: {stats['fps']:.2f}")

    if not all_stats:
        return

    first_key = list(all_stats.keys())[0]
    total_frames = all_stats[first_key]["frame_count"]
    print(f"\n📈 CLAHE 效果比對統計 (總幀數: {total_frames}, 模型: {model_name}):")
    for label, s in all_stats.items():
        print(f"\n🔹 [{label}]")
        print(f"  - 效能表現: {s['fps']:.2f} FPS")
        print(f"  - 總計偵測到人臉次數: {s['total_faces']}")
        sorted_ids = sorted(s["identities"].items(), key=lambda x: x[1], reverse=True)
        print(f"  - 辨識分佈 (TOP 5):")
        for name, count in sorted_ids[:5]:
            print(f"    * {name}: {count} 次")

    print("\n📊 差異分析 (With CLAHE vs No CLAHE):")
    no_clahe_ids = all_stats["No CLAHE"]["identities"]
    clahe_ids = all_stats["With CLAHE"]["identities"]
    all_names = sorted(set(no_clahe_ids.keys()) | set(clahe_ids.keys()))
    for name in all_names:
        before = no_clahe_ids.get(name, 0)
        after = clahe_ids.get(name, 0)
        diff = after - before
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"  * {name}: No CLAHE={before} | With CLAHE={after} | 差異={diff_str}")

# ==========================================
# 多模型效能 PK 功能
# ==========================================
def benchmark_multi_model(models_list, video_path, mode='native'):
    print(f"\n🏆 開始多模型效能 PK ({mode} Mode): {video_path}")
    if not os.path.exists(video_path): return

    results = {}
    
    # 確保 Reset 追蹤變數
    global frame_counter, tracked_identities

    for model_name in models_list:
        print(f"\n⚙️ 正在測試模型: {model_name}")
        
        # Reset variables for each model
        frame_counter = 0
        tracked_identities = []
        
        # 初始化該模型
        # 若是 native 模式，不需要 YOLO
        load_yolo = (mode == 'yolo')
        app, yolo_model = init_insightface(model_name, load_yolo=load_yolo)
        known_embeddings, known_names, _ = load_known_faces(app)
        
        cap = VideoCaptureThreading(video_path)
        count = 0
        start = time.time()
        
        try:
            while True:
                frame = cap.read()
                if frame is None:
                    if not cap.running: break
                    time.sleep(0.001)
                    continue
                
                frame = crop_and_pad_center(frame)
                
                # 測試時關閉繪圖 (draw=False) 以取得純運算 FPS
                if mode == 'yolo':
                    process_yolo_pipeline(app, yolo_model, frame, known_embeddings, known_names, draw=False)
                else:
                    process_native_pipeline(app, frame, known_embeddings, known_names, draw=False)

                count += 1
                if count % 50 == 0:
                    elapsed = time.time() - start
                    print(f"\r  Running... {count} frames ({count/elapsed:.1f} FPS)", end="")
        except KeyboardInterrupt:
            pass
            
        total_time = time.time() - start
        avg_fps = count / total_time if total_time > 0 else 0
        cap.release()
        
        results[model_name] = avg_fps
        print(f"\n✅ 模型 {model_name} --> {avg_fps:.2f} FPS")

    print(f"\n🏁 最終排行榜 ({mode}):")
    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (m, fps) in enumerate(sorted_res):
        print(f"  {i+1}. {m}: {fps:.2f} FPS")

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
                
                frame = crop_and_pad_center(frame)
                
                if mode_name == "YOLO+Crop":
                    process_func(app, yolo_model, frame, known_embeddings, known_names, draw=False)
                else:
                    process_func(app, frame, known_embeddings, known_names, draw=False)

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
    if len(known_embeddings) > 0:
        print("\n請選擇執行模式:")
        print("1. [All Benchmark] 極速效能測試 (YOLO + Native)")
        print("2. [Visualize + Benchmark] 視覺化比對後進行效能測試 (Dual View)")
        print("3. [Accuracy Comparison] 多模型影片準確率比對 (Full Frame)")
        print("4. [YOLO Visualize] YOLO+Crop 視覺化並統計平均速度")
        print("5. [Native Visualize] Native Full Frame 視覺化並統計平均速度")
        print("6. [Model VS Visual] 雙模型即時視覺化對決 (Native Mode)")
        print("7. [Multi-Model Benchmark] 多模型 YOLO+Crop 效能 PK")
        print("8. [Multi-Model Benchmark] 多模型 Native Full 效能 PK")
        print("9. [CLAHE Impact] 比較 CLAHE 前後辨識次數 (buffalo_m + Native)")

        choice = input("請輸入選項 (1-9): ").strip()
        
        models_pk_list = ['my_arcface_pack', 'buffalo_m', 'my_arcface_pack_40', 'r50MS1MV3']

        if choice == '1':
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, run_mode='all')
        elif choice == '2':
            compare_faces(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, known_files, mode='all')
            benchmark_performance(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, run_mode='all')
        elif choice == '3':
            # 定義想要比較的模型列表
            models_to_test = models_pk_list
            evaluate_video_accuracy(models_to_test, TARGET_IMAGE_PATH)
        elif choice == '4':
            compare_faces(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, known_files, mode='yolo')
        elif choice == '5':
            compare_faces(app, yolo_model, TARGET_IMAGE_PATH, known_embeddings, known_names, known_files, mode='native')
        elif choice == '6':
            model_a = 'buffalo_m'  # 基準模型
            default_model = MODEL_NAME
            model_b = input(f"請輸入要比較的模型名稱 (預設: {default_model}): ").strip() or default_model
            visual_compare_models(model_a, model_b, TARGET_IMAGE_PATH)
        elif choice == '7':
            benchmark_multi_model(models_pk_list, TARGET_IMAGE_PATH, mode='yolo')
        elif choice == '8':
            benchmark_multi_model(models_pk_list, TARGET_IMAGE_PATH, mode='native')
        elif choice == '9':
            evaluate_clahe_impact(TARGET_IMAGE_PATH, model_name='buffalo_m')
        else:
            print("❌ 無效選項，結束程式")
    else:
        print("⚠️ 資料庫為空")