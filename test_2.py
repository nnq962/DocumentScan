import cv2
import numpy as np
from ultralytics import YOLO

def run_webcam_segmentation(model_path, confidence_threshold=0.5):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)  # Mở webcam (0 là cam mặc định)

    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi khi đọc frame từ webcam!")
            break

        height, width = frame.shape[:2]
        results = model(frame)
        result = results[0]

        overlay = frame.copy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()
        names = result.names

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

            for idx, (mask, conf, cls_id) in enumerate(zip(masks, confs, clses)):
                if conf < confidence_threshold:
                    continue  # Bỏ qua nếu thấp hơn ngưỡng tin cậy

                label_name = names[int(cls_id)]
                box = boxes[idx]
                x1, y1, x2, y2 = box.astype(int)
                label_text = f"{label_name} {conf:.2f}"

                color_box = (0, 255, 0) if label_name.lower() == "document" else (0, 0, 255)

                if label_name.lower() == "document":
                    # Resize mask và tô màu xanh lá cây
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    colored_mask = np.zeros_like(frame, dtype=np.uint8)
                    colored_mask[mask_binary == 1] = (0, 255, 0)  # Màu xanh lá
                    overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

                # Vẽ bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_box, 2)

                # Vẽ label bên trong box
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(overlay, (x1, y1), (x1 + tw + 6, y1 + th + 8), color_box, -1)
                cv2.putText(overlay, label_text, (x1 + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Hiển thị frame kết quả
        cv2.imshow("Webcam Segmentation", overlay)

        # Nhấn 'Q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy webcam với model:
run_webcam_segmentation("./yolo11s-seg.pt", confidence_threshold=0.6)