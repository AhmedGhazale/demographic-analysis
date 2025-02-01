# Backend

backend implementation using flask that perform AI analysis and communicate with frontend using web sockite

# Folder Structure

* **app.py**: implementation of the flask backend.
* **ai/**: contains ai pipeline implementation
    * **tflite_interface.py**: an abstract class that implement inference using tensorflow lite and has 2 virtual functions *preprocessing* and *postprocessing*.
    * **age_gender_model.py**: an implementation of the TFLiteModel abstract class with the preprocessing and postprocessing functions for the age/gender model.
    * **face_detector_model.py**: an implementation of the TFLiteModel abstract class with the preprocessing and postprocessing functions for YOLO v11 nano model.
    * **models/**: contains tensorflow lite models for age/gender and face detector.
        * **age_gender_float16.tflite**: age gender model quantized to float16 precision.
        * **face_detector_float16.tflite**: face detection model quantized to float16 precision.