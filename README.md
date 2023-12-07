# FaceDetectCam
It succinctly describes the project's purpose: real-time face detection using a webcam. Feel free to choose a name that aligns with your preferences or consider adding a creative touch that reflects the uniqueness of your face detection application.
import cv2

class FaceDetector:
    def __init__(self):
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

    def detect_faces(self):
        while True:
            # Capture a frame from the webcam
            ret, frame = self.cap.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Face Detection', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close the OpenCV window
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    face_detector = FaceDetector()
    face_detector.detect_faces()
