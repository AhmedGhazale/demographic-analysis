from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import cv2


class TFLiteModel(ABC):
    """Abstract base class for TFLite model inference."""

    def __init__(self, model_path):
        """
        Initialize TFLite interpreter and allocate tensors.

        Args:
            model_path: Path to the TFLite model file
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details for reference
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @abstractmethod
    def preprocess(self, input_data):
        """
        Virtual method for preprocessing input data (to be implemented by subclass).

        Args:
            input_data: Raw input data to be processed

        Returns:
            Processed input data as numpy array
        """
        pass

    @abstractmethod
    def postprocess(self, output_data):
        """
        Virtual method for postprocessing model output (to be implemented by subclass).

        Args:
            output_data: Raw model output tensor

        Returns:
            Processed result in desired format
        """
        pass

    def predict(self, input_data):
        """
        Perform complete inference pipeline: preprocess -> inference -> postprocess.

        Args:
            input_data: Raw input data to be processed

        Returns:
            Final processed output from postprocessing
        """
        # Preprocess input data
        processed_input = self.preprocess(input_data)

        # Ensure input type matches model expectation
        if processed_input.dtype != self.input_details[0]['dtype']:
            raise ValueError(f"Input dtype mismatch. Expected {self.input_details[0]['dtype']}, "
                             f"got {processed_input.dtype}")

        # Set input tensor and run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_input)
        self.interpreter.invoke()

        # Get output tensor
        output_data = [self.interpreter.get_tensor(output['index']) for output in self.output_details]

        # Postprocess and return final result
        return self.postprocess(output_data)


class ImageClassifier(TFLiteModel):
    """Example implementation for image classification"""

    def preprocess(self, image):
        # Example preprocessing: resize, normalize, expand dimensions
        processed = cv2.resize(image, (224, 224))
        processed = processed / 255.0
        return np.expand_dims(processed, axis=0).astype(np.float32)

    def postprocess(self, output_data):
        # Example postprocessing: softmax and get top class
        return output_data

if __name__ =='__main__':

    model = ImageClassifier("/home/ahmed/PycharmProjects/demographic_analysis/ai/age_gender/tf_model/model_float32.tflite")

    input = np.random.rand(224,224,3)
    print(model.pr)