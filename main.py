import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from keras.layers import InputLayer

data_dir = "dataset/"
cordena = data_dir + 'cordana/'
healthy = data_dir + 'healthy/'
pestalotiopsis = data_dir + 'pestalotiopsis/'
sigatoka = data_dir + 'sigatoka/'
cordena_1 = os.listdir(cordena)
healthy_1 = os.listdir(healthy)
pestalotiopsis_1 = os.listdir(pestalotiopsis)
sigatoka_1 = os.listdir(sigatoka)


def loading(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    return img[..., ::-1]


for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(loading(cordena + cordena_1[i]), cmap='gray')
    plt.suptitle("cordana", fontsize=20)
    plt.axis('off')

plt.show()

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(loading(healthy + healthy_1[i]), cmap='gray')
    plt.suptitle("healthy", fontsize=20)
    plt.axis('off')

plt.show()

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(loading(pestalotiopsis + pestalotiopsis_1[i]), cmap='gray')
    plt.suptitle("pestalotiopsis", fontsize=20)
    plt.axis('off')

plt.show()

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(loading(sigatoka + sigatoka_1[i]), cmap='gray')
    plt.suptitle("sigatoka", fontsize=20)
    plt.axis('off')

plt.show()

data = []
labels = []

for class_label in tqdm(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, class_label)
    
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        img = load_img(img_path, target_size=(69, 69))
        img_array = img_to_array(img)
        data.append(img_array)
        labels.append(class_label)

X = np.array(data)
y = np.array(labels)

# Convert class labels to numeric format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


def build_mobilenet():
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(69, 69, 3))

    x = mobilenet.output
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(4, activation='softmax', name='root')(x)

    # model
    model = Model(inputs=mobilenet.input, outputs=output)
    
    optimizer = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model
model = build_mobilenet()
history = model.fit(X_train, y_train, batch_size=32, epochs=25, verbose=1, validation_data=(X_test, y_test), shuffle=True)




def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

plot_history(history)

loss, accuracy = model.evaluate(X_train, y_train)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

train_features = model.predict(X_train)
test_features = model.predict(X_test)

np.save("features/train_features.npy", train_features)
np.save("features/test_features.npy", test_features)

train_features_flat = train_features.reshape(train_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)


lr_model = LogisticRegression(max_iter=2000)
lr_model.fit(train_features_flat, np.argmax(y_train, axis=1))

lr_model_path = "model/lr_model.pkl"
with open(lr_model_path, 'wb') as file:
    pickle.dump(lr_model, file)


lr_pred = lr_model.predict(test_features_flat)

accuracy_1 = accuracy_score(np.argmax(y_test, axis=1), lr_pred)
print(f"Ensemble Model Accuracy: {accuracy_1}")

classification_report_result = classification_report(np.argmax(y_test, axis=1), lr_pred)

print(f"Classification Report:\n{classification_report_result}")

cm=confusion_matrix(np.argmax(y_test, axis=1), lr_pred)

print(f"Classification Report:\n{cm}")

data="dataset/healthy/0_aug.jpeg"
sample_img = load_img(data, target_size=(69, 69))
sample_img_array = img_to_array(sample_img)
sample_img_array = np.expand_dims(sample_img_array, axis=0)
sample_img_features = model.predict(sample_img_array)
sample_img_features_flat = sample_img_features.reshape(1, -1)
sample_pred = int(np.round((lr_model.predict(sample_img_features_flat) + lr_model.predict(sample_img_features_flat)) / 2)[0])
predicted_class_label = label_encoder.classes_[sample_pred]
print(f"prediction: {predicted_class_label}")
  
plt.figure()
plt.imshow(sample_img)
plt.title(f"Predicted Class: {predicted_class_label}")
plt.axis('off')
plt.show()

