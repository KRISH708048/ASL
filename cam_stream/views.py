import cv2 as cv
import threading
import time
import numpy as np
import mediapipe as mp
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators import gzip
from keras.models import load_model

print("Starting the Django project")

@gzip.gzip_page
def index(request):
    try:
        cam = VideoCamera()
        hand = Hand()
        return StreamingHttpResponse(gen(cam, hand), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        return HttpResponse(f"<h1>Error: {e}</h1>")
    

class Hand(object):
    def __init__(self):
        self.mphand = mp.solutions.hands
        self.hand = self.mphand.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def process(self, frame):
        RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.hand.process(RGBframe)
        hand = result.multi_hand_landmarks
        if hand:
            h, w, c = frame.shape
            landmarks_pixel_coords = [(int(l.x * w), int(l.y * h))
                                      for l in hand[0].landmark]
            x, y, w, h = cv.boundingRect(np.array(landmarks_pixel_coords))
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def preprocessfunc(self, image):
        gray_scale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        print("gray shape:",gray_scale_image.shape)
        resized_gray_image = cv.resize(gray_scale_image, (28, 28))
        flat_img = resized_gray_image.flatten()
        flat_img = flat_img.reshape(1, 28, 28, 1)
        return flat_img

class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture(0)
        self.grab, self.frame = self.video.read()
        self.time = time.time()
        self.counter =0 

        # Load the Keras model when the class is initialized
        self.model = load_model("saved_models/final_model_cnn.h5")

        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    
    def get_frame(self, hand):
        image = self.frame
        print("shape:",image.shape)
        if self.counter == 8:
            processed_frame = hand.preprocessfunc(image)
            print("after pre shape:",processed_frame.shape)
        # Debug: Print the processed frame
        # print("Processed Frame:", processed_frame)
        
            output = self.model.predict(processed_frame)
        
        # output= prediction_output(processed_frame)
        # Debug: Print the raw model output
            print("Raw Model Output:", output)
        
            max_index = np.argmax(output)  # Get the index of the highest value
            if max_index >= 9:
                max_index = max_index + 1
            
            cv.putText(image, str(max_index), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            self.counter =0
        self.counter+=1
        
        hand.process(image)
        ptime = time.time()
        fps = 1 / (ptime - self.time)
        self.time = ptime
        
        # Debug: Print the FPS and other values
        print("FPS:", int(fps))
        cv.putText(image, str(int(fps)), (50, 120),
                   cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        _, img_jpeg = cv.imencode('.jpg', image)
        return img_jpeg.tobytes()

    def update(self):
        while True:
            self.grab, self.frame = self.video.read()

def gen(camera, hand):
    while True:
        frame = camera.get_frame(hand)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
