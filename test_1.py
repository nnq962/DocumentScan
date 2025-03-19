import cv2
import numpy as np
from ultralytics import YOLO

def visualize_segmentation_with_inner_label(img_path, model_path, save_path=None, confidence_threshold=0.5):
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    results = model(img)
    result = results[0]

    overlay = img.copy()
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy()
    names = result.names

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # (num_objects, 640, 480)

        for idx, (mask, conf, cls_id) in enumerate(zip(masks, confs, clses)):
            if conf < confidence_threshold:
                continue  # bỏ qua đối tượng không đạt ngưỡng

            label_name = names[int(cls_id)]
            box = boxes[idx]
            x1, y1, x2, y2 = box.astype(int)
            label_text = f"{label_name} {conf:.2f}"

            color_box = (0, 255, 0) if label_name.lower() == "document" else (0, 0, 255)

            if label_name.lower() == "document":
                # Resize mask về kích thước ảnh gốc và tô màu xanh lá cây
                mask_resized = cv2.resize(mask, (width, height))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask_binary == 1] = (0, 255, 0)  # xanh lá cây
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

            # Vẽ bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_box, 2)

            # Vẽ label bên trong box
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(overlay, (x1, y1), (x1 + tw + 6, y1 + th + 8), color_box, -1)
            cv2.putText(overlay, label_text, (x1 + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("Segmented with inside labels & threshold", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, overlay)
        print(f"Đã lưu kết quả tại: {save_path}")

visualize_segmentation_with_inner_label(
    img_path="./IMG_1324.png", 
    model_path="./yolo11s-seg.pt", 
    save_path=None,
    confidence_threshold=0.6   # Bạn có thể chỉnh ngưỡng tùy ý
)