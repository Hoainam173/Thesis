import cv2 
import numpy as np
import time
from datetime import datetime
import os
import pandas as pd

# === Thư mục lưu kết quả học thuật ===
SAVE_DIR = "fiber_diameter"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, "fiber_measurements.csv")

# === Thông số hiệu chuẩn (cần xác định trước bằng thước chuẩn) ===
SCALE_MM_PER_PX = 0.1  # ví dụ: 1 pixel = 0.1 mm

# === Điều kiện lọc sợi theo yêu cầu ===
MIN_LENGTH_MM = 100.0  # Chiều dài tối thiểu: 100 mm
MAX_DIAMETER_MM = 5.0   # Đường kính tối đa: 5 mm
MIN_AREA_PX = 200      # Vùng diện tích tối thiểu để loại bỏ nhiễu

# === Khởi tạo camera Rapoo ===
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print(" Không mở được camera Rapoo. Kiểm tra kết nối USB.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# === Hàm xử lý ảnh và đo từng sợi ===
def process_fibers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold adaptif cho điều kiện sáng không đều
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphology loại nhiễu và làm liền vùng sợi
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fibers_data = []
    presentation = frame.copy()

    fiber_id = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:  # lọc nhiễu nhỏ
            continue

        # 1. Đo chiều dài (chu vi contour/2)
        perimeter = cv2.arcLength(c, True)
        length_mm = perimeter * SCALE_MM_PER_PX / 2

        # 2. Tính độ dày (đường kính trung bình)
        mask = np.zeros_like(clean)
        cv2.drawContours(mask, [c], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Kiểm tra để tránh lỗi chia cho 0 nếu không có pixel nào trong dist > 0
        if np.count_nonzero(dist > 0) == 0:
            thickness_mm = 0.0
        else:
            thickness_px = 2 * np.mean(dist[dist > 0])
            thickness_mm = thickness_px * SCALE_MM_PER_PX

        # 3. LỌC THEO ĐIỀU KIỆN (Áp dụng các yêu cầu mới)
        if length_mm < MIN_LENGTH_MM:
            # Sợi không đủ dài (dưới 100mm)
            continue
        if thickness_mm > MAX_DIAMETER_MM or thickness_mm == 0.0:
            # Sợi quá dày (trên 5mm) hoặc không xác định được đường kính
            continue
        
        # Nếu đạt điều kiện, tiếp tục xử lý
        fiber_id += 1
        x, y, w, h = cv2.boundingRect(c)
        
        # Ghi nhận thông tin
        fibers_data.append([fiber_id, x, y, w, h, length_mm, thickness_mm])

        # Vẽ kết quả lên ảnh trình bày
        cv2.rectangle(presentation, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(presentation,
                    f"ID:{fiber_id} L:{length_mm:.2f}mm D:{thickness_mm:.2f}mm",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return thresh, presentation, fibers_data


# === Chương trình chính ===
print(" Hệ thống đo sợi polyme (5 phút/lần)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Không đọc được hình từ camera.")
            break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Xử lý ảnh
        thresh, presentation, data = process_fibers(frame)

        # === BỔ SUNG HIỂN THỊ ĐƯỜNG KÍNH TRUNG BÌNH LÊN IMAGESHOW ===
        if data:
            # Lấy cột đường kính (vị trí thứ 6 trong mỗi list con)
            diameters = [item[6] for item in data]
            avg_diameter = np.mean(diameters)
            
            # Tạo chuỗi hiển thị
            avg_text = f"AVG D: {avg_diameter:.3f} mm ({len(data)} fibers)"
            
            # Vẽ nền (background) cho chữ dễ nhìn hơn
            (text_w, text_h), baseline = cv2.getTextSize(avg_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(presentation, (10, 10), (10 + text_w + 5, 10 + text_h + 10), (255, 255, 255), -1)
            
            # Vẽ chữ màu đỏ
            cv2.putText(presentation,
                        avg_text,
                        (15, 15 + text_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Lưu ảnh các bước cho báo cáo
        raw_path = os.path.join(SAVE_DIR, f"raw_{timestamp}.jpg")
        thresh_path = os.path.join(SAVE_DIR, f"threshold_{timestamp}.jpg")
        pres_path = os.path.join(SAVE_DIR, f"presentation_{timestamp}.jpg")
        cv2.imwrite(raw_path, frame)
        cv2.imwrite(thresh_path, thresh)
        cv2.imwrite(pres_path, presentation)

        # Lưu từng ROI sợi (ảnh zoom)
        for (fid, x, y, w, h, length_mm, dia_mm) in data:
            roi = frame[y:y+h, x:x+w]
            zoom_path = os.path.join(SAVE_DIR, f"zoom_fiber{fid}_{timestamp}.jpg")
            # Kiểm tra kích thước ROI trước khi resize
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                cv2.imwrite(zoom_path, cv2.resize(roi, (400, 400), interpolation=cv2.INTER_CUBIC))
            else:
                print(f"Cảnh báo: ROI sợi ID {fid} có kích thước không hợp lệ để zoom.")

        # Ghi dữ liệu CSV
        df = pd.DataFrame(data, columns=["Fiber_ID", "x", "y", "w", "h", "Length_mm", "Diameter_mm"])
        df["Timestamp"] = timestamp
        # Thêm header nếu file chưa tồn tại
        header_exists = os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0
        df.to_csv(LOG_FILE, mode='a', index=False, header=not header_exists)

        print(f"[{timestamp}] Đã lưu {len(data)} sợi thỏa mãn điều kiện. (Xem thư mục: {SAVE_DIR})")

        # Hiển thị
        cv2.imshow("Fiber Presentation", presentation)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            print(" Kết thúc chương trình.")
            break

        # Chờ 5 phút (300 giây)
        time.sleep(300)

except KeyboardInterrupt:
    print(" Dừng thủ công (Ctrl+C).")
finally:
    cap.release()
    cv2.destroyAllWindows()