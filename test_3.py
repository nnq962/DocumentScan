import cv2
import numpy as np
from ultralytics import YOLO

def visualize_mask_contour_crop_and_straighten(img_path, model_path, save_path=None, confidence_threshold=0.5):
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    results = model(img)
    result = results[0]

    final_mask = np.zeros((height, width), dtype=np.uint8)

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

                kernel = np.ones((5, 5), np.uint8)
                mask_smooth = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
                mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel)

                final_mask = cv2.bitwise_or(final_mask, mask_smooth)

    result_img = cv2.bitwise_and(img, img, mask=final_mask)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Tìm hình chữ nhật bao ngoài theo mask dạng nghiêng
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Tính toán để xoay hình cho thẳng
        angle = rect[2]
        if angle < -45:
            angle += 90

        center = rect[0]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(result_img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

        # Cắt ảnh theo bounding box sau khi xoay
        rotated_mask = cv2.warpAffine(final_mask, M, (width, height), flags=cv2.INTER_NEAREST)
        contours_rotated, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_rotated:
            x, y, w, h = cv2.boundingRect(max(contours_rotated, key=cv2.contourArea))
            cropped = rotated[y:y+h, x:x+w]

            cv2.imshow("Straightened Document", cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if save_path:
                cv2.imwrite(save_path, cropped)
                print(f"Đã lưu kết quả tại: {save_path}")

visualize_mask_contour_crop_and_straighten(
    img_path="./IMG_1324.png", 
    model_path="./yolo11s-seg.pt", 
    save_path=None,
    confidence_threshold=0.6
)
