# CC transfer code
# number of tunable parameters are similar to QCNN model

from typing import final
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import tensorflow_hub as hub
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(xtr, ytr), (xte, yte) = fashion_mnist.load_data()
xtr = xtr.reshape(-1, 28, 28, 1).astype("float32") / 255.0
xte = xte.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# dictionary which encoding has what number of parameters
encoding_to_parameter = {'Amplitude': 256, 'Angle': 8, 'Amplitude_Hybrid_4': 32, 'Amplitude_Hybrid_2': 16,
                         'Angular_Hybrid_4': 30, 'Angular_Hybrid_2': 12, 'Angle_compact': 16}
emb = 'Amplitude'
input_layer_parameter = encoding_to_parameter[emb]

# Pre-train CNN model
model_start = keras.Sequential()
model_start.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='layer1'))
model_start.add(layers.MaxPooling2D((2, 2), name='layer2'))
model_start.add(layers.Conv2D(64, (3, 3), activation='relu', name='layer3'))
model_start.add(layers.MaxPooling2D((2, 2), name='layer4'))
model_start.add(layers.Conv2D(64, (3, 3), activation='relu', name='layer5'))
model_start.add(layers.Flatten(name='layer6'))
# this layer have to same number of input parameters of QCNN
model_start.add(layers.Dense(input_layer_parameter, activation='relu', name='layer7'))
model_start.add(layers.BatchNormalization())
model_start.add(layers.Dense(30, activation='relu', name='layer8'))
model_start.add(layers.Dense(10, activation='softmax', name='fin'))

filepath = "./save/pretrain.h5"

# Callback Earlystopping (cb) :
# avoid overfitting during trainning
# Callback ModelCheck (ck) :
# monitoring training weight and save best weight or last weight

cb = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='auto', restore_best_weights=True)
ck = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='accuracy', verbose=0, save_best_only=True, mode='auto')

print("Basic pretrain model")
print(model_start.summary())

model_start.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
model_start.fit(xtr, ytr, epochs=5, callbacks=[ck])


model_start.evaluate(xte, yte, verbose=2)

# Freeze layers which means NO CHANGE during training
model = keras.models.load_model("./save/pretrain.h5")
model.trainable = False
for layer in model.layers:
    assert layer.trainable is False
    layer.trainable = False

# 10000 data for fine-tuning
(xtr, ytr), (xte, yte) = mnist.load_data()
xtr = xtr.reshape(-1, 28, 28, 1).astype("float32") / 255.0
xte = xte.reshape(-1, 28, 28, 1).astype("float32") / 255.0
dataset = (xtr, xte, ytr, yte)


def data_process(dataset, classes=[0, 1]):
    """
    input : dataset, class
    output : X_train, X_test, Y_train, Y_test
    three possible classes
    1. 'odd_even'
    2. '>4'
    3. other class by list
    """
    x_train, x_test, y_train, y_test = dataset
    if classes == 'odd_even':
        odd = [1, 3, 5, 7, 9]
        X_train = x_train
        X_test = x_test
        Y_train = [1 if y in odd else 0 for y in y_train]
        Y_test = [1 if y in odd else 0 for y in y_test]

    elif classes == '>4':
        greater = [5, 6, 7, 8, 9]
        X_train = x_train
        X_test = x_test
        Y_train = [1 if y in greater else 0 for y in y_train]
        Y_test = [1 if y in greater else 0 for y in y_test]

    else:
        # find index of y which class is same as classes[0] or classes[1]
        # note that | is bitwise or operation
        x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
        x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))

        # remain X and Y data set which class is same as classes[0] or classes[1]
        X_train, X_test = x_train[x_train_filter_01], x_test[x_test_filter_01]
        Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]
        Y_train = [1 if y == classes[0] else 0 for y in Y_train]
        Y_test = [1 if y == classes[0] else 0 for y in Y_test]
    return X_train, X_test, Y_train, Y_test


# Training conditions
batch_size = 50
steps = 200

# fine-tuning CNN model
cc_transfer_tup = ('1D_', '2D_')

# Fine-tuning CNN and write accuracies to file
for classes in [[0, 1], [2, 3], [8, 9]]:
    # 2115 data were used to evaluate accuracy for 0,1 classification
    # 2042 data were used to evaluate accuracy for 2,3 classification
    # 1983 data were used to evaluate accuracy for 8,9 classification
    xtr, xte, ytr, yte = data_process(dataset=dataset, classes=classes)
    xtr = xtr[:batch_size * steps]
    ytr = ytr[:batch_size * steps]
    ytr = np.array(ytr)
    yte = np.array(yte)
    for method in cc_transfer_tup:
        f = open('save/Transfer_result_' + method + 'CC' + str(classes) + '.txt', 'a')
        for _ in range(10):

            # Split model into pre-train model layers and the other layers
            base_inputs = model.layers[0].input
            base_outputs = model.layers[-3].output

            # prefered trainable parameter ~ 51

            # Amplitude encoding 256 to (16*16) image 2D CNN method
            # parameter 76
            if method == '2D_':
                base_outputs = layers.Reshape((16, 16, 1), input_shape=(256,))(base_outputs)
                final_outputs = layers.Conv2D(2, (3, 3), activation='relu', name='layer8')(base_outputs)  # parameters 76
                final_outputs = layers.MaxPooling2D((2, 2), name='layer9')(final_outputs)
                final_outputs = layers.Conv2D(2, (3, 3), activation='relu', name='layer10')(final_outputs)  # parameters 76
                final_outputs = layers.MaxPooling2D((2, 2), name='layer11')(final_outputs)
                final_outputs = layers.Flatten(name='layer12')(final_outputs)
                final_outputs = layers.Dense(2, activation='softmax', name='layer13')(final_outputs)

            # 1D CNN method : parameter 64
            if method == '1D_':
                base_outputs = layers.Reshape((256, 1), input_shape=(256,))(base_outputs)
                final_outputs = layers.Conv1D(1, 5, activation='relu', name='layer8', strides=3)(base_outputs)  # parameter 64
                final_outputs = layers.MaxPooling1D(3, name='layer9')(final_outputs)
                final_outputs = layers.Flatten(name='layer10')(final_outputs)
                final_outputs = layers.Dense(2, activation='softmax', name='layer11')(final_outputs)

            new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)
            print("New model")
            print(new_model.summary())

            new_model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
            History = new_model.fit(xtr, ytr, batch_size=batch_size, epochs=1)

            test_loss, test_acc = new_model.evaluate(xte, yte, verbose=2)
            print("Test accuracy: ", test_acc)
            f.write("Accuracy: \n")
            f.write(str(test_acc))
            f.write("\n")
        f.flush()
