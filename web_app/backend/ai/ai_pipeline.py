import cv2

from .face_detector_model import FaceDetector
from .age_gender_model import AgeGenderEstimator



class AIPipeline:


    def __init__(self, face_detector_path, age_gender_model_path):


        self.face_detector = FaceDetector(face_detector_path)
        self.age_gender_model = AgeGenderEstimator(age_gender_model_path)


    def crop_face(self, image, box):

        x,y,w,h  = box
        w = max(w, h)
        h = max(w,h)

        x1 = x - w // 2
        y1 = y - h // 2

        x2 = x + w // 2
        y2 = y + h // 2

        x1 = max(0,x1)
        y1 = max(0,y1)

        x2 = min(image.shape[1],x2)
        y2 = min(image.shape[0],y2)

        crop = image[y1: y2, x1: x2, :]

        return crop

    def run(self, image):

        boxes, scores = self.face_detector.predict(image)

        ages_genders = []

        for box in boxes:

            face = self.crop_face(image, box)
            age_gender = self.age_gender_model.predict(face)

            ages_genders.append(age_gender)



        # drawing results

        for box, age_gender in zip(boxes, ages_genders):

            age, gender = age_gender
            x,y,w,h = box

            x1=  x - w//2
            y1 = y-h//2

            x2 = x1 + w
            y2 = y1 + h
            cv2.rectangle(image, (x1,y1),(x2,y2), (255,0,0),2)
            cv2.putText(image, str(age) + " "+gender, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


        return image


if __name__ == "__main__":

    pipline = AIPipeline('models/face_detector_float16.tflite', "models/age_gender_float16.tflite")



    img = cv2.imread("/media/ahmed/data/new_backup/Home/me/IMG_4771.JPG")
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    res = pipline.run(rgb)

    cv2.cvtColor(res,cv2.COLOR_BGR2RGB)


    cv2.imshow('win',res)

    cv2.waitKey(-1)