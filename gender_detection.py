import cv2
import numpy as np

# Paths to the pre-trained models
gender_deploy = "gender_deploy.prototxt"  # Gender deploy file (prototxt)
gender_model = "gender_net.caffemodel"   # Gender model file (caffemodel)

age_deploy = "age_deploy.prototxt"        # Age deploy file (prototxt)
age_model = "age_net.caffemodel"          # Age model file (caffemodel)

# Initialize the gender and age detection models
gender_net = cv2.dnn.readNet(gender_model, gender_deploy)
age_net = cv2.dnn.readNet(age_model, age_deploy)

# Gender labels (Male or Female)
gender_list = ['Male', 'Female']

# Age labels (Age groups)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Preprocessing parameters
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

while True:
    success, frame = cam.read()  # Capture frame from webcam
    if not success:
        print("Failed to grab frame")
        break

    # Resize the frame to the required input size for gender and age detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Gender Detection
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()  # Run the gender model on the frame
    gender = gender_list[np.argmax(gender_preds)]  # Get the predicted gender

    # Age Detection
    age_net.setInput(blob)
    age_preds = age_net.forward()  # Run the age model on the frame
    age = age_list[np.argmax(age_preds)]  # Get the predicted age group

    # Display the predicted gender and age on the frame
    cv2.putText(frame, f"Gender: {gender}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Age: {age}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame with gender and age labels
    cv2.imshow("Gender and Age Detection", frame)

    # Break the loop if 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Release the camera and close any OpenCV windows
cam.release()
cv2.destroyAllWindows()
