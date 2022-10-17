# Import libraries
import cv2
import glob
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Dense, Dropout, TimeDistributed
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import CSVLogger
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dtype = np.dtype('B') # Unsigned byte
# dtype = np.dtype('i4') # 32-bit signed integer

# Create an image dataset
images_list = glob.glob('C:/AI_InSAR/dataset/train3/*.bin')
labels_list = glob.glob('C:/AI_InSAR/dataset/Mask/mask3/*.npy')

images = np.zeros((len(images_list), 100, 100, 1), dtype)
labels = np.zeros((len(labels_list), 100, 100, 1), dtype)

num = -1
for file in images_list:
    num = num + 1
    # print(file)
    try:
        with open(file, "rb") as f:
            numpy_data = np.fromfile(f, dtype)
            numpy_dataq = np.reshape(numpy_data, (100, 100, 1))
            numpy_datai = numpy_dataq.astype(int)
            images[num, :, :, :] = numpy_datai
    except IOError:
        print('Error While Opening the file!')
images = images.reshape(15, 100, 100, 100, 1)
images = np.transpose(images, (1, 0, 2, 3, 4))
print(images.shape)

# Creating a label dataset
num = -1
for file in labels_list:
    num = num + 1
    try:
        with open(file, "rb") as f:
            # numpy_data = np.fromfile(f, dtype)
            numpy_data = np.load(f)
            numpy_dataq = np.reshape(numpy_data, (100, 100, 1))
            numpy_datai = numpy_dataq.astype(int)
            labels[num, :, :, :] = numpy_datai
    except IOError:
        print('Error While Opening the file!')

N = 15
result = np.vstack([labels]*N)
# print(result.shape)
# 15 - time, 100 - No. of interferograms, 100 - x, 100 - y, 1 - channel
labels = result.reshape(15, 100, 100, 100, 1)
labels = np.transpose(labels, (1, 0, 2, 3, 4))
# print(labels[0, 1, 80:100, 80:100, 0])
print(labels.shape)

# Dataset split to train and test
from sklearn.model_selection import train_test_split
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.4)
# print(images_train.shape, images_test.shape)
# print(labels_train.shape, labels_test.shape)

# Create a LSTM RNN model
model = Sequential()
model.add(ConvLSTM2D(16, 3, strides = 1, padding='same', dilation_rate = 2, return_sequences=True, input_shape = (15, 100, 100, 1)))
model.add(BatchNormalization())
# model.add(ConvLSTM2D(16, 3, strides = 1, padding='same', dilation_rate = 2, return_sequences=True, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(ConvLSTM2D(32, 3, strides = 1, padding='same', dilation_rate = 2, return_sequences=True, activation='relu'))
model.add(BatchNormalization())
model.add(ConvLSTM2D(64, 3, strides = 1, padding='same', dilation_rate = 2, return_sequences=True, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(100, activation='sigmoid')))
model.add(TimeDistributed(Dense(1, activation='sigmoid'))) # softmax or relu or tanh or sigmoid
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # Could be also as loss='mse' and optimizer='adam'and metrics=[RootMeanSquaredError(), MeanAbsoluteError()]
# model.summary()

csv_logger = CSVLogger("my_history_50.csv", append=True)
# history = model.fit(np.array(images), np.array(labels), batch_size=32, epochs=50, verbose=2, validation_split=0.2, callbacks=(csv_logger)) # With data split randomly
# history = model.fit(np.array(images_train), np.array(labels_train), batch_size=2, epochs=5, verbose=2, validation_data=(images_test, labels_test)) # With manual data separation

# scores = model.evaluate(images_test, labels_test, verbose=2)
# print(scores)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

# np.save('my_history_50.npy', history.history)
# history = np.load('my_history.npy', allow_pickle='TRUE').item()
# print(history)

# Graph of Accuracy and Loss
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# Save the architecture of the model + the weights + the training configuration + the state of the optimizer
# save_model(model, "model_50.h5")
# print("Saved model to disk")

# Path to the test data
images_list_test = glob.glob('C:/AI_InSAR/dataset/test_dataset/test1/*.bin')
labels_list_test = glob.glob('C:/AI_InSAR/dataset/test_dataset/mask1/*.npy')

images_test = np.zeros((len(images_list_test), 100, 100, 1), dtype)
labels_test = np.zeros((len(labels_list_test), 100, 100, 1), dtype)

# Create an image shape for test
num = -1
for file in images_list_test:
    num = num + 1
    try:
        with open(file, "rb") as f:
            numpy_data = np.fromfile(f, dtype)
            numpy_dataq = np.reshape(numpy_data, (100, 100, 1))
            numpy_datai = numpy_dataq.astype(int)
            images_test[num, :, :, :] = numpy_datai
    except IOError:
        print('Error While Opening the file!')

images1test = images_test.reshape(15, 1, 10000)
images1test = np.transpose(images1test, (1, 0, 2))

images1test = images_test.reshape(15, 1, 100, 100, 1)
images1test = np.transpose(images1test, (1, 0, 2, 3, 4))
print(images1test.shape)

# Creating a label shape for test
num = -1
for file in labels_list_test:
    num = num + 1
    try:
        with open(file, "rb") as f:
            numpy_data = np.load(f)
            numpy_dataq = np.reshape(numpy_data, (100, 100, 1))
            numpy_datai = numpy_dataq.astype(int)
            labels_test[num, :, :, :] = numpy_datai
    except IOError:
        print('Error While Opening the file!')

N = 15
result = np.vstack([labels_test]*N)
labels1test = result.reshape(15, 1, 100, 100, 1)
labels1test = np.transpose(labels1test, (1, 0, 2, 3, 4))
print(labels1test.shape)

# Load the model
model.load_weights('model_2_10.h5')
model.summary()

# predicted = model.predict(images1test)
# predicted = np.round(predicted, 1).astype(np.int32)
# print(predicted.shape)

scores = model.evaluate(images1test, images1test, verbose=1)
print('Scores: Test loss:', scores[0], 'Test accuracy:', scores[1])

y_predict = np.argmax(model.predict(images1test), axis=1)
y_predict2 = np.argmax(model.predict(images1test), axis=1)
# pass in the input and set the the learning phase to 0
print(y_predict.shape)
# print(y_predict.dtype)
# print(y_predict[0, 0:30, 0:30, 0])

for i in range(y_predict.shape[1]):
    for j in range(y_predict.shape[2]):
        if y_predict[0, i, j, 0] > 6:
            y_predict[0, i, j, 0] = 1
        else:
            y_predict[0, i, j, 0] = 0

# Visualization of the result
f, axarr = plt.subplots(1, 4)
axarr[0].imshow(images1test[0, 1, :, :, 0], interpolation='nearest', cmap = plt.cm.gray)
axarr[1].imshow(labels1test[0, 0, :, :, 0], interpolation='nearest', cmap = plt.cm.gray)
axarr[2].imshow(y_predict2[0, :, :, 0], interpolation='nearest', cmap = plt.cm.binary_r)
axarr[3].imshow(y_predict[0, :, :, 0], interpolation='nearest', cmap = plt.cm.binary_r)

axarr[0].set_title("Image")
axarr[1].set_title("PS Mask")
axarr[2].set_title("Prediction")
axarr[3].set_title("Treshold")

axarr[0].set_xlabel('pixel')
axarr[0].set_ylabel('pixel')
axarr[1].set_xlabel('pixel')
# axarr[1].set_ylabel('pixel')
axarr[2].set_xlabel('pixel')
# axarr[2].set_ylabel('pixel')
axarr[3].set_xlabel('pixel')
# axarr[3].set_ylabel('pixel')

plt.show()
plt.axis('off')




