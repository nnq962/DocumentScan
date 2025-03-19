import cv2
import numpy as np
from ultralytics import YOLO

def visualize_segmentation_mask_with_quadrilateral(img_path, model_path, save_path=None, confidence_threshold=0.5):
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    results = model(img)
    result = results[0]

    overlay = img.copy()

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()
        names = result.names

        for mask, conf, cls_id in zip(masks, confs, clses):
            if conf < confidence_threshold:
                continue

            label_name = names[int(cls_id)]

            if label_name.lower() == "document":
                mask_resized = cv2.resize(mask, (width, height))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # Tìm contour
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        # Tìm hình tứ giác gần đúng từ contour
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        if len(approx) == 4:
                            quadrilateral = approx.reshape(-1, 2)
                            cv2.polylines(overlay, [quadrilateral], isClosed=True, color=(0, 0, 255), thickness=3)

                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask_binary == 1] = (0, 255, 0)
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)

    cv2.imshow("Segmented Mask with Quadrilateral", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, overlay)
        print(f"Đã lưu kết quả tại: {save_path}")

visualize_segmentation_mask_with_quadrilateral(
    img_path="./IMG_1324.png", 
    model_path="./yolo11s-seg.pt", 
    save_path=None,
    confidence_threshold=0.6
)
