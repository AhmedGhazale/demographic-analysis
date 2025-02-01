from .tflite_interface import TFLiteModel
import cv2
import numpy as np



class AgeGenderEstimator(TFLiteModel):
    """Example implementation for image classification"""

    def preprocess(self, image):
        # Example preprocessing: resize, normalize, expand dimensions
        processed = cv2.resize(image, (224, 224))
        processed = processed / 255.0

        processed -= np.array([0.485, 0.456, 0.406])
        processed /=np.array( [0.229, 0.224, 0.225])

        return np.expand_dims(processed, axis=0).astype(np.float32)

    def postprocess(self, output_data):
        # Example postprocessing: softmax and get top class

        genders = ['male','female']

        gender = genders[int(output_data[0].item()) > .5]
        age = int(output_data[1].item())
        return age, gender


if __name__ == "__main__":

    model = AgeGenderEstimator("/home/ahmed/PycharmProjects/demographic_analysis/ai/age_gender/model.tflite")


    img = cv2.imread("/web_app/backend/ai/crop.jpg")

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    print(model.predict(img))
