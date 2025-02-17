import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image #type:ignore
from tensorflow.keras.models import Sequential, load_model #type:ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
import cv2

# 🔹 Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Root directory
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODEL_PATH = os.path.join(BASE_DIR, 'mymodel.h5')
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# 🔹 Model Parameters
IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 10

# 🔹 Check if model exists
if os.path.exists(MODEL_PATH):
    print(" Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print(" Training new model...")

    # 🔹 Build Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 🔹 Data Augmentation
    train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # 🔹 Load Dataset
    training_set = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    test_set = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')

    # 🔹 Train Model
    model.fit(training_set, epochs=EPOCHS, validation_data=test_set)

    # 🔹 Save Model
    model.save(MODEL_PATH)
    print(f" Model saved as {MODEL_PATH}")

# 🔹 Function for Image Prediction
def predict_mask(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "No Mask" if prediction > 0.5 else "Mask"

# 🔹 Live Face Mask Detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

if face_cascade.empty():
    print(" Error: Could not load Haar Cascade classifier.")
    exit()

print(" Starting Live Face Mask Detection... Press 'Q' to exit.")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print(" Error: Could not read frame.")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        temp_path = os.path.join(BASE_DIR, 'temp.jpg')
        cv2.imwrite(temp_path, face_img)  # Save temp image for prediction
        result = predict_mask(temp_path)
        os.remove(temp_path)  # Remove temp file after prediction

        color = (0, 255, 0) if result == "Mask" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Face Mask Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Live Detection Stopped.")
