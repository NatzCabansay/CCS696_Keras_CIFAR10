import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras import layers

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(f"X_train = {X_train.shape}")
print(f"y_train = {y_train.shape}")
print(f"X_test = {X_test.shape}")
print(f"y_test = {y_test.shape}")

lbls = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

fig, axes = plt.subplots(5,5, figsize = (10,10))
axes = axes.ravel()
for i in np.arange(0, 5*5):
    idx = np.random.randint(0, len(X_train))
    axes[i].imshow(X_train[idx,1:])
    lbl_idx = int(y_train[idx])
    axes[i].set_title(lbls[lbl_idx], fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

classes, counts = np.unique(y_train, return_counts=True)
plt.figure()
plt.barh(lbls, counts)
plt.title('Class distribution in training set')

X_train = X_train/255.0
X_test = X_test/255.0

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(X_train, y_train_cat, test_size = 0.2, random_state = 42)
batch_size = 64
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip = True)
train_generator = data_generator.flow(X_TRAIN, Y_TRAIN, batch_size)

INPUT_SHAPE = (32, 32, 3)
model = Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=(3,3), input_shape = INPUT_SHAPE, activation = 'relu', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=16, kernel_size=(3,3), input_shape = INPUT_SHAPE, activation = 'relu', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=32, kernel_size=(3,3), input_shape = INPUT_SHAPE, activation = 'relu', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), input_shape = INPUT_SHAPE, activation = 'relu', padding = 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_generator, epochs=10, validation_data=(X_VAL, Y_VAL),)

plt.figure(figsize=(12,8))

plt.subplot(4,2,1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_Loss')
plt.title('Loss')
plt.legend()

plt.subplot(4,2,2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Loss')
plt.legend()

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
con = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lbls)
fig, ax = plt.subplots(figsize=(10,10))
con = con.plot(xticks_rotation = 'vertical', ax=ax, cmap='summer')

import random
plt.figure()
idx = random.randint(0, len(X_test))
im = X_test[idx]
plt.imshow(im)

pred_t = np.argmax(model.predict(im.reshape(1, 32, 32, 3)))
print(f"Our model predicts that image {idx} is {lbls[pred_t]}")
plt.show()
