from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import cv2
import os

ROOT_DIR = "E:/research/"
DATA_DIR = "Wafer"
IMAGES = "Images"
MASKS = "Masks"
LABELS = "Labels"
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
NUMBER_EPOCHS = 300

classMapping = {
    "[0 0 0 0 0 0 0 0]":0,
    "[1 0 0 0 0 0 0 0]":1,
    "[0 1 0 0 0 0 0 0]":2,
    "[0 0 1 0 0 0 0 0]":3,
    "[0 0 0 1 0 0 0 0]":4,
    "[0 0 0 0 1 0 0 0]":5,
    "[0 0 0 0 0 1 0 0]":6,
    "[0 0 0 0 0 0 1 0]":7,
    "[0 0 0 0 0 0 0 1]":8,
    "[1 0 1 0 0 0 0 0]":9,
    "[1 0 0 1 0 0 0 0]":10,
    "[1 0 0 0 1 0 0 0]":11,
    "[1 0 0 0 0 0 1 0]":12,
    "[0 1 1 0 0 0 0 0]":13,
    "[0 1 0 1 0 0 0 0]":14,
    "[0 1 0 0 1 0 0 0]":15,
    "[0 1 0 0 0 0 1 0]":16,
    "[0 0 1 0 1 0 0 0]":17,
    "[0 0 1 0 0 0 1 0]":18,
    "[0 0 0 1 1 0 0 0]":19,
    "[0 0 0 1 0 0 1 0]":20,
    "[0 0 0 0 1 0 1 0]":21,
    "[1 0 1 0 1 0 0 0]":22,
    "[1 0 1 0 0 0 1 0]":23,
    "[1 0 0 1 1 0 0 0]":24,
    "[1 0 0 1 0 0 1 0]":25,
    "[1 0 0 0 1 0 1 0]":26,
    "[0 1 1 0 1 0 0 0]":27,
    "[0 1 1 0 0 0 1 0]":28,
    "[0 1 0 1 1 0 0 0]":29,
    "[0 1 0 1 0 0 1 0]":30,
    "[0 1 0 0 1 0 1 0]":31,
    "[0 0 1 0 1 0 1 0]":32,
    "[0 0 0 1 1 0 1 0]":33,
    "[1 0 1 0 1 0 1 0]":34,
    "[1 0 0 1 1 0 1 0]":35,
    "[0 1 1 0 1 0 1 0]":36,
    "[0 1 0 1 1 0 1 0]":37
    }

# data loader

class waferSegDataLoader(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batchSize, imgSize, inputImgPaths, targetImgPaths, labelsImgPaths):
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.inputImgPaths = inputImgPaths
        self.targetImgPaths = targetImgPaths
        self.labelsImgPaths = labelsImgPaths

    def __len__(self):
        return len(self.targetImgPaths) // self.batchSize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchSize
        batchInputImgPaths = self.inputImgPaths[i : i + self.batchSize]
        batchTargetImgPaths = self.targetImgPaths[i : i + self.batchSize]
        batchLabelsImgPaths = self.labelsImgPaths[i : i + self.batchSize]

        x = np.zeros((self.batchSize,) + self.imgSize + (3,), dtype="float32")
        y = np.zeros((self.batchSize,) + self.imgSize + (1,), dtype="float32")
        z = np.zeros((self.batchSize,) + self.imgSize + (38,), dtype="float32")
        for j, (input_image, input_mask, input_label) in enumerate(zip(batchInputImgPaths, batchTargetImgPaths, batchLabelsImgPaths)):
            img = np.load(input_image)
            x[j] = img.astype("float32")
            
            #img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
            msk = np.load(input_mask)
            msk = msk / 255.0
            
            y[j] = msk.astype("float32")
            lbl = classMapping[str(input_label)]
            print(lbl)
            lbl = tf.keras.utils.to_categorical(lbl, num_classes=38)
            print(lbl)
            z[j] = lbl

        return x, (y, z)

inputImgPaths = sorted([os.path.join(ROOT_DIR, DATA_DIR, IMAGES, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, IMAGES))])
targetImgPaths = sorted([os.path.join(ROOT_DIR, DATA_DIR, MASKS, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, MASKS))])
labelsImgPaths = sorted([os.path.join(ROOT_DIR, DATA_DIR, LABELS, x) for x in os.listdir(os.path.join(ROOT_DIR, DATA_DIR, LABELS))])

#trainInputImgPaths, testInputImgPaths, trainTargetImgPaths, testTargetImgPaths = train_test_split(inputImgPaths, targetImgPaths, test_size = 0.2, random_state = 42)
trainInputImgPaths = inputImgPaths[:int(len(inputImgPaths)*0.8)]
testInputImgPaths = inputImgPaths[int(len(inputImgPaths)*0.8):]
trainTargetImgPaths = targetImgPaths[:int(len(targetImgPaths)*0.8)]
testTargetImgPaths = targetImgPaths[int(len(targetImgPaths)*0.8):]
trainLabelsImgPaths = labelsImgPaths[:int(len(labelsImgPaths)*0.8)]
testLabelsImgPaths = labelsImgPaths[int(len(labelsImgPaths)*0.8):]

trainGen = waferSegDataLoader(batchSize = BATCH_SIZE, imgSize = IMG_SIZE, inputImgPaths = trainInputImgPaths, targetImgPaths = trainTargetImgPaths, labelsImgPaths = trainLabelsImgPaths)
testGen = waferSegDataLoader(batchSize = BATCH_SIZE, imgSize = IMG_SIZE, inputImgPaths = testInputImgPaths, targetImgPaths = testTargetImgPaths, labelsImgPaths = testLabelsImgPaths)

def conv_layer(input_layer, conv_channels, kernel_size = (3,3), pool_stride = (2,2), dropout_rate = 0.2, padding = 'same', activation = 'relu'):        
    #print(input_layer.shape)
    layer_1 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(input_layer)
    #print(layer_1.shape)
    layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_3 = tf.keras.layers.Dropout(dropout_rate)(layer_2)
    layer_4 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(layer_3)
    layer_5 = tf.keras.layers.BatchNormalization()(layer_4)
    layer_6 = tf.keras.layers.MaxPool2D(pool_stride)(layer_5)
    
    return layer_1,layer_6

def gap_conv_layer(input_layer, conv_channels, kernel_size = (3,3), pool_stride = (2,2), dropout_rate = 0.2, padding = 'same', activation = 'relu'):        
    #print(input_layer.shape)
    layer_1 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(input_layer)
    #print(layer_1.shape)
    layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_3 = tf.keras.layers.Dropout(dropout_rate)(layer_2)
    layer_4 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(layer_3)
    layer_5 = tf.keras.layers.BatchNormalization()(layer_4)
    layer_6 = tf.keras.layers.AveragePooling2D(pool_stride)(layer_5)
    
    return layer_1,layer_6

# Function for a single resolution convolution operation at the terminal position

def terminal_conv_layer(input_layer, conv_channels, kernel_size = (3,3), dropout_rate = 0.2, padding = 'same', activation = 'relu'):
    
    layer_1 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(input_layer)
    layer_2 = tf.keras.layers.BatchNormalization()(layer_1)
    layer_3 = tf.keras.layers.Dropout(dropout_rate)(layer_2)
    layer_4 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(layer_3)
    layer_5 = tf.keras.layers.BatchNormalization()(layer_4)
    
    return layer_5

# Function for a single resolution transpose convolution operation

def transpose_conv_layer(input_layer, skip_layer, conv_channels, kernel_size = (3,3), transpose_kernel_size = (2,2), dropout_rate = 0.2, padding = 'same', activation = 'relu', transpose_strides = (2,2)):
    #print(input_layer.shape)
    layer_1 = tf.keras.layers.Conv2DTranspose(conv_channels, transpose_kernel_size, strides = transpose_strides, padding = padding )(input_layer)
    #print(layer_1.shape, skip_layer.shape)
    layer_2 = tf.keras.layers.concatenate([layer_1, skip_layer], axis = 3)
    
    layer_3 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(layer_2)
    layer_4 = tf.keras.layers.BatchNormalization()(layer_3)
    layer_5 = tf.keras.layers.Dropout(dropout_rate)(layer_4)
    layer_6 = tf.keras.layers.Conv2D(conv_channels, kernel_size, activation = activation, padding = padding, kernel_initializer = 'he_normal')(layer_5)
    layer_7 = tf.keras.layers.BatchNormalization()(layer_6)    
    
    return layer_7

def modified_unet_model(height, width, image_channels):
    
    inputs = tf.keras.layers.Input((height, width, image_channels))
    normalized_inputs = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

    S1,D1 = conv_layer(normalized_inputs, 16)
    S2,D2 = conv_layer(D1, 32)
    S3,D3 = conv_layer(D2, 64)
    S4,D4 = conv_layer(D3, 128)
    S5,D5 = gap_conv_layer(D4, 256)

    T1 = terminal_conv_layer(D5, 512)

    gap = tf.keras.layers.GlobalAveragePooling2D()(T1)
    d1 = tf.keras.layers.Dense(64, activations="relu")(gap)
    d2 = tf.keras.layers.Dense(38, activations="softmax", name="classification")(d1)

    U1 = transpose_conv_layer(T1, S5, 256)
    U2 = transpose_conv_layer(U1, S4, 128)
    U3 = transpose_conv_layer(U2, S3, 64)
    U4 = transpose_conv_layer(U3, S2, 32)
    U5 = transpose_conv_layer(U4, S1, 16)
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation = "sigmoid", name="segmentations")(U5)

    model = tf.keras.models.Model(inputs = [inputs], outputs = [outputs, d2])
    
    model.summary()
    
    return model

def diceCoef(y_true, y_pred):   
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)    
    y_pred_f = K.flatten(y_pred)    
    intersection = K.sum(y_true_f * y_pred_f)    
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)

def diceCoefLoss(y_true, y_pred):
    return 1.0 - diceCoef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + diceCoefLoss(y_true, y_pred)
    return loss

def train_the_model(trainGen, testGen, epochs):
    
    unet_model = modified_unet_model(224, 224, 3)
    
    unet_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = {"segmentation":bce_dice_loss, "classification":"categorical_crossentropy"}, metrics = ["accuracy", diceCoef], loss_weights = {"segmentation": 1.0, "classification": 1.0})
    callbacks = [
      tf.keras.callbacks.ModelCheckpoint('wafer_unet.h5', monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = False)
    ]
    history = unet_model.fit(trainGen, validation_data = testGen, epochs = epochs, verbose = 1, callbacks=callbacks)
    
    return history, unet_model

history, trained_model = train_the_model(trainGen, testGen, 500)
trained_model.load_weights("wafer_unet.h5")
#print(trained_model.evaluate(testGen))

import matplotlib.pyplot as plt

for i, (img, msk) in enumerate(testGen):
    fig, ax = plt.subplots(1, 2)
    img = np.load(inputImgPaths[33033])
    msk = np.load(targetImgPaths[33033])
    ax[0].imshow(msk[:, :, 0])
    pred = trained_model.predict(np.expand_dims(img, 0))
    ax[1].imshow(pred[0][:, :, 0])
    plt.show()
    break