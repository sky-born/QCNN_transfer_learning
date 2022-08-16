# Implementation of Quantum circuit training procedure
from struct import pack

from numpy import double
from QOSF import QCNN_circuit
from QOSF import Hierarchical_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp


def square_loss(labels, predictions):
    loss = 0
    for label, p in zip(labels, predictions):
        loss = loss + (label - p) ** 2

    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    loss = 0
    delta = 1e-7
    for label, p in zip(labels, predictions):
        # label is 1 or -1, p is probability numpy array with length 2
        if label == -1:
            label = 0
        c_entropy = label * (anp.log(p[label] + delta)) + (1 - label) * anp.log(1 - p[1 - label] + delta)
        loss = loss + c_entropy
    return -1 * loss


def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN' or circuit == 'QCNN_general_pooling':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
    elif circuit == 'Hierarchical':
        predictions = [Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]

    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)

    return loss


# Circuit training parameters
# step size
steps = 200
# steps = 400 # steps for underfitting
# steps = 30
learning_rate = 0.01
batch_size = 50


def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN':
        total_params = U_params * 3 + 2 * 3
    elif circuit == 'Hierarchical':
        total_params = U_params * 7
    elif circuit == 'QCNN_general_pooling':
        total_params = U_params * 3 + 6 * 3

    params = np.random.randn(total_params, requires_grad=True)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    # opt = qml.AdamOptimizer(stepsize=learning_rate)
    loss_history = []
    # best_cost = None
    # best_params = None

    for it in range(steps):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        params, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn),
            params)  # basic cost fuction
        loss_history.append(cost_new)

        # save best cost parameters -> can overfitting
        # if best_cost == None:
        #     best_cost = cost_new
        #     best_params = params
        # if best_cost > cost_new:
        #     best_cost = cost_new
        #     best_params = params

        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
    # return best paragrams
    return loss_history, params
