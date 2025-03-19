import cv2
import numpy as np
from ultralytics import YOLO

class SegmentationVisualizer:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        results = self.model(frame)
        result = results[0]

        cropped_results = []

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy()
            names = result.names

            for mask, conf, cls_id in zip(masks, confs, clses):
                if conf < self.confidence_threshold:
                    continue

                label_name = names[int(cls_id)]

                if label_name.lower() == "document":
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)

                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            epsilon = 0.02 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)

                            if len(approx) == 4:
                                quadrilateral = approx.reshape(-1, 2)

                                rect = np.zeros((4, 2), dtype="float32")
                                s = quadrilateral.sum(axis=1)
                                rect[0] = quadrilateral[np.argmin(s)]
                                rect[2] = quadrilateral[np.argmax(s)]

                                diff = np.diff(quadrilateral, axis=1)
                                rect[1] = quadrilateral[np.argmin(diff)]
                                rect[3] = quadrilateral[np.argmax(diff)]

                                (tl, tr, br, bl) = rect
                                widthA = np.linalg.norm(br - bl)
                                widthB = np.linalg.norm(tr - tl)
                                maxWidth = max(int(widthA), int(widthB))

                                heightA = np.linalg.norm(tr - br)
                                heightB = np.linalg.norm(tl - bl)
                                maxHeight = max(int(heightA), int(heightB))

                                dst = np.array([
                                    [0, 0],
                                    [maxWidth - 1, 0],
                                    [maxWidth - 1, maxHeight - 1],
                                    [0, maxHeight - 1]
                                ], dtype="float32")

                                M = cv2.getPerspectiveTransform(rect, dst)
                                warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

                                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                                sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
                                sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
                                filtered = cv2.adaptiveThreshold(
                                    sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
                                )
                                filtered_colored = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

                                cropped_results.append(filtered_colored)

        return cropped_results if cropped_results else [frame]

    def inference_image(self, image_path, save_dir="output"):
        img = cv2.imread(image_path)
        results = self.process_frame(img)

        for idx, result_img in enumerate(results):
            cv2.imshow(f"Document {idx+1}", result_img)
            save_path = f"{save_dir}/document_{idx+1}.png"
            cv2.imwrite(save_path, result_img)
            print(f"Đã lưu: {save_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def inference_webcam(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.process_frame(frame)

            for idx, res in enumerate(results):
                cv2.imshow(f"Webcam Document {idx+1}", res)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage:
visualizer = SegmentationVisualizer(model_path="./yolo11s-seg.pt", confidence_threshold=0.9)
# visualizer.inference_image(image_path="./IMG_0356.png")
visualizer.inference_webcam()
