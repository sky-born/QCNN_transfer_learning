# CQ transfer code
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import tensorflow_hub as hub
import pennylane as qml
from QOSF import QCNN_circuit
from QOSF import unitary
from QOSF import embedding
import numpy as np
from QOSF import Training
from QOSF.Benchmarking import accuracy_test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(xtr, ytr), (xte, yte) = fashion_mnist.load_data()
xtr = xtr.reshape(-1, 28, 28, 1).astype("float32") / 255.0
xte = xte.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# dictionary which encoding has what number of parameters
encoding_to_parameter = {'Amplitude': 256, 'Angle': 8, 'Amplitude_Hybrid_4': 32, 'Amplitude_Hybrid_2': 16,
                         'Angular_Hybrid_4': 30, 'Angular_Hybrid_2': 12, 'Angle_compact': 16}
emb = 'Angle'
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

# Split model into pre-train model layers and the other layers
base_inputs = model.layers[0].input
base_outputs = model.layers[-3].output
new_model = keras.Model(inputs=base_inputs, outputs=base_outputs)
print("New model")
print(new_model.summary())

# 10000 data for fine-tuning
(xtr, ytr), (xte, yte) = mnist.load_data()
xtr = xtr.reshape(-1, 28, 28, 1).astype("float32") / 255.0
xte = xte.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Original training and test data are transformed by pre-trained model
X_train = new_model.predict(xtr)
X_test = new_model.predict(xte)
dataset = (X_train, X_test, ytr, yte)


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


def Transfer_QCNN(dataset, classes, Unitary, U_num_param, emb, circuit, cost_fn):
    """
    input : dataset, classes, Unitary, U_num_param, emb, circuit, cost_fn
    output : best_trained_params_list
    Fine-tuning QCNN to train classical-to-quantum transfer learning.
    QCNN model is defined by Unitary ansatz.
    This function writes trained models parameters, accuracy, and loss history to a file.
    definition of input paramters are listed below
    dataset: composed of train and test data
    classes: binary classification target
    Unitary: QCNN convolution and pooling ansatz
    U_num_param: number of trainable parameters of unitary ansatz.
    emb: encoding method. (Amplitude encoding in this work)
    circuit: 'QCNN' or 'QCNN_general_pooling' which depends on pooling
    cost_fn: cost function, 'cross_entropy' or 'mse'
    """
    U = Unitary
    U_params = U_num_param
    best_trained_params_list = []

    # class [0,1] => X train: 12665, X test: 2115
    # class [2,3] => X train: 12089, X test: 2042
    # class [8,9] => X train: 11800, X test: 1983
    X_train, X_test, Y_train, Y_test = data_process(dataset=dataset, classes=classes)
    Embedding = emb
    f = open('save/Transfer_result_' + emb + U + str(classes) + '.txt', 'a')
    trained_params_list = []
    accuracy_list = []
    for n in range(10):

        print("\n")
        print("Loss History for " + circuit + " circuits, " + U + " " + emb)

        loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit, cost_fn)
        predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
        accuracy = accuracy_test(predictions, Y_test, cost_fn, binary=False)
        print("Accuracy for " + U + " " + emb + " :" + str(accuracy))

        trained_params_list.append(trained_params)
        accuracy_list.append(accuracy)

        f.write("Trained Parameters: \n")
        f.write(str(trained_params))
        f.write("\n")
        f.write("Accuracy: \n")
        f.write(str(accuracy))
        f.write("\n")
        f.write("Loss history: \n")
        f.write(str(loss_history))
        f.write("\n")
        f.flush()

    index = accuracy_list.index(max(accuracy_list))
    best_trained_params_list.append(trained_params_list[index])

    f.close()
    return best_trained_params_list


# Quantum circuit can have 'QCNN' and 'QCNN_general_pooling' model
# circuit = 'QCNN'
# circuit = 'QCNN_general_pooling'

# Unitary and unitary parameters are written in unitary.py

# cost_fn can select 'cross_entropy' or 'mse'
cost_fn = 'cross_entropy'

Unitary_dict = {'U_TTN': 2, 'U_5': 10, 'U_6': 10, 'U_9': 2, 'U_13': 6, 'U_14': 6, 'U_15': 4, 'U_SO4': 6, 'U_SU4': 15,
                'U_TTN_R': 2, 'U_5_R': 10, 'U_6_R': 10, 'U_9_R': 2, 'U_13_R': 6, 'U_14_R': 6, 'U_15_R': 4, 'U_SO4_R': 6, 'U_SU4_R': 15,
                'U_SU4_no_pooling': 15, 'U_ansatz_10': 9, 'Con_Z': 0, 'Con_Z_R': 0}

# Unitary_dict = {'U_TTN' : 2, 'U_5' : 10, 'U_6' : 10, 'U_9' : 2, 'U_13' : 6, 'U_14' : 6, 'U_15' : 4, 'U_SO4' : 6, 'U_SU4' : 15}

# Bench mark codes are sorted by pooling.
# To run these code, uncomment the codes.

# ZX pooling
# circuit = 'QCNN'
# for Unitary in ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'Con_Z']:
#     U_num_param = Unitary_dict[Unitary]
#     for classes in [[0, 1], [2, 3], [8, 9]]:
#         Transfer_QCNN(dataset, classes, Unitary, U_num_param, emb, circuit, cost_fn)

# General pooling
# circuit = 'QCNN_general_pooling'
# for Unitary in ['U_TTN_R', 'U_5_R', 'U_6_R', 'U_9_R', 'U_13_R', 'U_14_R', 'U_15_R', 'U_SO4_R', 'U_SU4_R','Con_Z_R']:
#     U_num_param = Unitary_dict[Unitary]
#     for classes in [[0, 1], [2, 3], [8, 9]]:
#         Transfer_QCNN(dataset, classes, Unitary, U_num_param, emb, circuit, cost_fn)

# Trivial pooling
circuit = 'QCNN'
for Unitary in ['U_SU4_no_pooling', 'U_ansatz_10']:
    U_num_param = Unitary_dict[Unitary]
    for classes in [[0, 1], [2, 3], [8, 9]]:
        Transfer_QCNN(dataset, classes, Unitary, U_num_param, emb, circuit, cost_fn)
