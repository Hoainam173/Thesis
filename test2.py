import cv2
import numpy as np
import os
import pandas as pd
import time
from datetime import datetime
import glob
import threading
import tkinter as tk
from tkinter import ttk

# THƯ MỤC LƯU 
SAVE_DIR = "fiber_diameter"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, "fiber_measurements.csv")

# CAMERA 
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Không mở được camera Rapoo. Kiểm tra kết nối USB.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# HIỆU CHUẨN CAMERA BẰNG CHESSBOARD 
CHESSBOARD_DIR = "chessboard_images"
CHESSBOARD_SIZE = (9,6)   # số góc bên trong
SQUARE_SIZE_MM = 5        # mm

def calibrate_camera(chessboard_dir, board_size=(9,6), square_size_mm=5):
    objp = np.zeros((board_size[0]*board_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1,2)
    objp *= square_size_mm

    objpoints = []
    imgpoints = []

    images = glob.glob(f'{chessboard_dir}/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, board_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(100)
    cv2.destroyAllWindows()

    if len(objpoints) < 2:
        print("Không đủ ảnh chessboard hợp lệ! Sử dụng SCALE_MM_PER_PX mặc định 0.1mm/pixel.")
        return None, None, 0.1  # mm/pixel mặc định

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    pixel_distance = np.linalg.norm(imgpoints[0][0][0] - imgpoints[0][1][0])
    scale_mm_per_px = square_size_mm / pixel_distance
    print("✅ Camera calibrated successfully.")
    print(f"Estimated SCALE_MM_PER_PX: {scale_mm_per_px:.4f} mm/pixel")
    return mtx, dist, scale_mm_per_px

camera_matrix, dist_coeffs, SCALE_MM_PER_PX = calibrate_camera(CHESSBOARD_DIR, CHESSBOARD_SIZE, SQUARE_SIZE_MM)

# ĐIỀU KIỆN LỌC SỢI 
MIN_LENGTH_MM = 100.0
MAX_DIAMETER_MM = 5.0
MIN_AREA_PX = 200

# GUI BÁO CÁO 
root = tk.Tk()
root.title("Fiber Quality Diagnosis")
root.geometry("480x250")
root.resizable(False, False)

title_label = tk.Label(root, text="🔬 Polymer Fiber Quality Diagnosis", font=("Arial", 14, "bold"))
title_label.pack(pady=10)

frame = ttk.Frame(root)
frame.pack(pady=5)

avg_label = ttk.Label(frame, text="Độ dày trung bình: --- mm", font=("Arial", 12))
min_label = ttk.Label(frame, text="Độ dày nhỏ nhất: --- mm", font=("Arial", 12))
max_label = ttk.Label(frame, text="Độ dày lớn nhất: --- mm", font=("Arial", 12))
count_label = ttk.Label(frame, text="Số sợi hợp lệ: ---", font=("Arial", 12))
quality_label = ttk.Label(root, text="Đang phân tích...", font=("Arial", 12, "bold"), foreground="blue")

avg_label.pack(anchor="w", pady=2)
min_label.pack(anchor="w", pady=2)
max_label.pack(anchor="w", pady=2)
count_label.pack(anchor="w", pady=2)
quality_label.pack(pady=10)

def update_gui(avg_dia, min_dia, max_dia, count):
    avg_label.config(text=f"Độ dày trung bình: {avg_dia:.3f} mm")
    min_label.config(text=f"Độ dày nhỏ nhất: {min_dia:.3f} mm")
    max_label.config(text=f"Độ dày lớn nhất: {max_dia:.3f} mm")
    count_label.config(text=f"Số sợi hợp lệ: {count}")

    deviation = max_dia - min_dia
    if deviation < 0.5:
        text = "Sợi đồng đều, đạt chuẩn chất lượng."
        color = "green"
    elif deviation < 1.0:
        text = "Sợi tương đối ổn định."
        color = "orange"
    else:
        text = "Sợi dao động lớn – cần kiểm tra máy kéo."
        color = "red"
    quality_label.config(text=text, foreground=color)

# HÀM XỬ LÝ ẢNH 
def process_fibers(frame):
    if camera_matrix is not None:
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fibers_data = []
    presentation = frame.copy()
    fiber_id = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        perimeter = cv2.arcLength(c, True)
        length_mm = perimeter * SCALE_MM_PER_PX / 2
        mask = np.zeros_like(clean)
        cv2.drawContours(mask, [c], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if np.count_nonzero(dist>0)==0:
            thickness_mm=0.0
        else:
            thickness_px = 2*np.mean(dist[dist>0])
            thickness_mm = thickness_px*SCALE_MM_PER_PX
        if length_mm<MIN_LENGTH_MM:
            continue
        if thickness_mm>MAX_DIAMETER_MM or thickness_mm==0.0:
            continue
        fiber_id +=1
        x,y,w,h = cv2.boundingRect(c)
        fibers_data.append([fiber_id,x,y,w,h,length_mm,thickness_mm])
        cv2.rectangle(presentation,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(presentation,
                    f"ID:{fiber_id} L:{length_mm:.2f}mm D:{thickness_mm:.2f}mm",
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    return thresh, presentation, fibers_data

# VÒNG LẶP CHÍNH TRONG THREAD RIÊNG 
def fiber_measure_loop():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không đọc được hình từ camera.")
                break
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            thresh, presentation, data = process_fibers(frame)
            
            if data:
                diameters = [item[6] for item in data]
                avg_diameter = np.mean(diameters)
                min_diameter = np.min(diameters)
                max_diameter = np.max(diameters)
                root.after(0, update_gui, avg_diameter, min_diameter, max_diameter, len(data))
            
            # Lưu ảnh và CSV như cũ
            raw_path = os.path.join(SAVE_DIR,f"raw_{timestamp}.jpg")
            thresh_path = os.path.join(SAVE_DIR,f"threshold_{timestamp}.jpg")
            pres_path = os.path.join(SAVE_DIR,f"presentation_{timestamp}.jpg")
            cv2.imwrite(raw_path,frame)
            cv2.imwrite(thresh_path,thresh)
            cv2.imwrite(pres_path,presentation)

            for (fid,x,y,w,h,length_mm,dia_mm) in data:
                roi = frame[y:y+h, x:x+w]
                zoom_path = os.path.join(SAVE_DIR,f"zoom_fiber{fid}_{timestamp}.jpg")
                if roi.shape[0]>0 and roi.shape[1]>0:
                    cv2.imwrite(zoom_path, cv2.resize(roi,(400,400), interpolation=cv2.INTER_CUBIC))
                else:
                    print(f"Cảnh báo: ROI sợi ID {fid} không hợp lệ để zoom.")

            df = pd.DataFrame(data,columns=["Fiber_ID","x","y","w","h","Length_mm","Diameter_mm"])
            df["Timestamp"]=timestamp
            header_exists = os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE)>0
            df.to_csv(LOG_FILE, mode='a', index=False, header=not header_exists)

            print(f"[{timestamp}] Đã lưu {len(data)} sợi hợp lệ. Xem thư mục: {SAVE_DIR}")
            cv2.imshow("Fiber Presentation",presentation)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                print("Kết thúc chương trình.")
                break
            time.sleep(300)
    except KeyboardInterrupt:
        print("Dừng thủ công (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# KHỞI ĐỘNG THREAD 
threading.Thread(target=fiber_measure_loop, daemon=True).start()

root.mainloop()
