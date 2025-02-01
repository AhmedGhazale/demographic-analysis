from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from ai.ai_pipeline import AIPipeline




ai_pipline = AIPipeline(face_detector_path="ai/models/face_detector_float16.tflite",
                        age_gender_model_path="ai/models/age_gender_float16.tflite")


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")



def process_frame(img):
    # Example processing: Edge detection
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 100, 200)
    return img
    # return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('frame')
def handle_frame(data):
    try:
        # Convert base64 string to image
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process frame
        # processed_img = process_frame(img)
        processed_img = ai_pipline.run(img)

        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        emit('processed_frame', f"data:image/jpeg;base64,{jpg_as_text}")

    except Exception as e:
        print(f"Error processing frame: {str(e)}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True,allow_unsafe_werkzeug=True )