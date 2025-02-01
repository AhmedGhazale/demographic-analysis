from .tflite_interface import TFLiteModel
import cv2
import numpy as np
from typing import Tuple, Union, List


class FaceDetector(TFLiteModel):

    def __init__(self, model_path):

        super().__init__(model_path)
        self.in_width = 640
        self.in_height = 640
        self.conf = .5
        self.iou = .5

        self.pad = None
        self.img = None

    def letterbox(self, img: np.ndarray, new_shape: Tuple = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top / img.shape[0], left / img.shape[1])

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed.

        Returns:
            Tuple[np.ndarray, Tuple[float, float]]: A tuple containing:
                - The preprocessed image (np.ndarray).
                - A tuple of two float values representing the padding applied (top/bottom, left/right).
        """
        self.img = img

        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[..., ::-1][None]  # N,H,W,C for TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        self.pad = pad
        return img / 255

    def postprocess(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            img (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output of the model.
            pad (Tuple[float, float]): Padding used by letterbox.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        outputs = outputs[0]
        pad = self.pad
        img = self.img

        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        outputs[:, :4] *= max(img.shape)

        outputs = outputs.transpose(0, 2, 1)[0]

        # outputs[..., 0] -= outputs[..., 2] / 2
        # outputs[..., 1] -= outputs[..., 3] / 2

        scores = outputs[..., 4]
        boxes = outputs[..., :4]

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()


        return np.astype(boxes[indices], np.int32), scores[indices]


if __name__ == "__main__":
    model = FaceDetector("/home/ahmed/PycharmProjects/demographic_analysis/ai/face_detector/runs/detect/train2/weights/best_saved_model/best_float16.tflite")

    img = cv2.imread("/media/ahmed/data/new_backup/Home/me/IMG_4771.JPG")
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    boxes, scores = model.predict(rgb)


    for box in boxes:


        x,y,w,h  = box
        w = max(w, h)
        h = max(w,h)
        x1 = x-w//2
        y1 = y-h//2

        cv2.rectangle(img, (x1, int(y1)), (int(x1 + w), int(y1 + h)), (255,0,0), 2)


    crop = img[y1: y1 + h, x1: x1 + w,:]

    cv2.imwrite("crop.jpg",crop)
    cv2.imshow('1',img)
    cv2.imshow('2',crop)
    cv2.waitKey(-1)