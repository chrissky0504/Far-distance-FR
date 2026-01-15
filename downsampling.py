import cv2

def resize_video(input_path, output_path, width=640, height=640):
    cap = cv2.VideoCapture(input_path)
    
    # 取得原始影片的 FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 設定影片編碼格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 初始化輸出物件
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 強制縮放到 640x640
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    resize_video('test4.mp4', 'output_640x640.mp4')
