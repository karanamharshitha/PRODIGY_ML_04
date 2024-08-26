pip install tensorflow keras opencv-python numpy matplotlib scikit-learn

import cv2
import os


cap = cv2.VideoCapture(0)


gestures = ['thumbs_up', 'thumbs_down', 'fist', 'palm', 'peace']
for gesture in gestures:
    os.makedirs(f'dataset/{gesture}', exist_ok=True)


for gesture in gestures:
    print(f'Collecting data for {gesture}')
    count = 0
    while count < 200:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  
        cv2.putText(frame, f'Collecting {gesture} - Image {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            img_name = f'dataset/{gesture}/{count}.jpg'
            cv2.imwrite(img_name, frame)
            print(f'Saved {img_name}')
            count += 1
        elif key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical


img_size = 64

def load_data(data_path, gestures):
    X = []
    y = []
    for idx, gesture in enumerate(gestures):
        gesture_dir = os.path.join(data_path, gesture)
        for img_name in os.listdir(gesture_dir):
            img_path = os.path.join(gesture_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1, img_size, img_size, 1)  # Add channel dimension
    X = X / 255.0  # Normalize
    y = to_categorical(y, num_classes=len(gestures))
    return X, y

gestures = ['thumbs_up', 'thumbs_down', 'fist', 'palm', 'peace']
X, y = load_data('dataset', gestures)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(gestures), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (img_size, img_size))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = normalized_frame.reshape(1, img_size, img_size, 1)

    prediction = model.predict(reshaped_frame)
    gesture = gestures[np.argmax(prediction)]

    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

model.save('hand_gesture_model.h5')