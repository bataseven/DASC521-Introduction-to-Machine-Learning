import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

reader = pd.read_csv('hw02_data_set_images.csv', header=None)
data = reader.values
reader = pd.read_csv('hw02_data_set_labels.csv', header=None)
labels = reader.values

labels.shape = (195,)

classes = np.array([1, 2, 3, 4, 5])

training_data = np.zeros((125, 320), dtype=int)
training_label = np.empty((125,), dtype=int)
testing_data = np.zeros((70, 320), dtype=int)
testing_label = np.empty((70,), dtype=int)

testing_index = 0
training_index = 0

for i in range(195):

    if i % 39 < 25:

        training_data[training_index] = data[i, :]
        if labels[i] == 'A':
            training_label[training_index] = 1
        elif labels[i] == 'B':
            training_label[training_index] = 2
        elif labels[i] == 'C':
            training_label[training_index] = 3
        elif labels[i] == 'D':
            training_label[training_index] = 4
        else:
            training_label[training_index] = 5

        training_index = training_index + 1

    else:

        testing_data[testing_index] = data[i, :]

        if labels[i] == 'A':
            testing_label[testing_index] = 1
        elif labels[i] == 'B':
            testing_label[testing_index] = 2
        elif labels[i] == 'C':
            testing_label[testing_index] = 3
        elif labels[i] == 'D':
            testing_label[testing_index] = 4
        else:
            testing_label[testing_index] = 5

        testing_index = testing_index + 1

np.random.seed(1)

K = 5
N = 320

W = np.random.uniform(low=-0.01, high=0.01, size=(N, K))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))


def sigmoid(x, W, w0):  # W - 320 5  x - 125 320
    return 1 / (1 + np.exp(-(x.dot(W) + w0)))


def calculate_loss(truth_matrix, prediction_matrix):
    return 0.5 * ((truth_matrix - prediction_matrix) ** 2).sum()


def calculate_w(x, truth_matrix, prediction_matrix):
    term1 = (truth_matrix - prediction_matrix) * prediction_matrix
    term2 = 1 - prediction_matrix
    chain_rule_eq = term1 * term2
    return W + eta * x.T.dot(chain_rule_eq)


def calculate_w0(truth_matrix, prediction_matrix):
    term1 = (truth_matrix - prediction_matrix) * prediction_matrix
    term2 = 1 - prediction_matrix
    chain_rule_eq = term1 * term2
    return w0 + eta * chain_rule_eq.sum(axis=0)


def create_confusion_matrix(truth_labels, predictions):
    conf_matrix = np.zeros((classes.size, classes.size), dtype=int)
    for i in range(len(truth_labels)):
        conf_matrix[predictions[i]][truth_labels[i] - 1] += 1
    return conf_matrix


eta = 0.01
epsilon = 1e-3
max_iteration = 1000
iteration = 1

label_indexes = training_label - 1
train_label_matrix = np.eye(classes.size, dtype=int)[label_indexes]

loss_values = np.zeros((max_iteration,), dtype=float)


while 1:
    prediction_matrix = sigmoid(training_data, W, w0)
    loss_values[iteration - 1] = calculate_loss(train_label_matrix, prediction_matrix)

    W = calculate_w(training_data, train_label_matrix, prediction_matrix)
    w0 = calculate_w0(train_label_matrix, prediction_matrix)

    iteration += 1
    if iteration >= max_iteration:
        break

plt.plot(loss_values)
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.show()

training_predictions = np.argmax(sigmoid(training_data, W, w0), axis=1)
confusion_matrix_1 = create_confusion_matrix(training_label, training_predictions)

print('Training Confusion Matrix:')
print(str(confusion_matrix_1) + '\n')

testing_predictions = np.argmax(sigmoid(testing_data, W, w0), axis=1)
confusion_matrix_2 = create_confusion_matrix(testing_label, testing_predictions)

print('Testing Confusion Matrix:')
print(str(confusion_matrix_2) + '\n')
