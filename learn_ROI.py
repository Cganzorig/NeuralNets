import numpy as np
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from keras.utils import to_categorical

from illustrate import illustrate_results_ROI

def evaluate_architecture(model, X, Y):
    accuracy = model.evaluate(X, Y, verbose=1)
    return accuracy


def predict_hidden(dataset):
    #Nomalize the data
    normalizer = preprocessing.Normalizer().fit(dataset)
    dataset = normalizer.transform(dataset)

    X = dataset[:,:3]
    Y = dataset[:,3:]

    model = load_model('best_model_Q3.h5')

    result = model.predict_classes(X)

    #one-hot encode
    categorical = to_categorical(result)

    return categorical


def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    classes = {}

    #Compute the weights for each label
    counts = [0, 0, 0, 0]
    for sample in dataset:
        for i in range(4):
            if(sample[i + 3] == 1):
                counts[i] += 1

    maximum = max(counts)
    weights = {0: maximum/counts[0], 1: maximum/counts[1], 2: maximum/counts[2], 3: maximum/counts[3]}
    print(weights)

    #Data preparation
    np.random.shuffle(dataset)
    length = len(dataset)

    test_dataset = dataset[:int(length/10),:]

    X_test = test_dataset[:, :3]
    Y_test = test_dataset[:, 3:]

    training_dataset = dataset[:int(9*(length/10)),:]

    X = training_dataset[:, :3]
    Y = training_dataset[:, 3:]

    #For saving the architecture during parameter tuning
    best_performance = 0
    best_model = []

    #Create and Evaluate model
    kf = KFold(n_splits=10)

    #Try these values for the parameters
    neurons = [16, 64, 128]
    layers = [1, 2, 3]

    for neuron in neurons:
        for layer in layers:
            for train, validation in kf.split(X, Y): #Cross validation
                model = Sequential()
                model.add(Dense(4, input_shape=(3,), activation='relu'))

                for i in range(layer):
                    model.add(Dense(neuron, activation='relu'))

                model.add(Dense(4, activation='softmax'))

                Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
                model.compile(Adam, loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(X[train], Y[train], batch_size=10, epochs=10, class_weight = weights, shuffle=True, verbose = 1)

                performance = evaluate_architecture(model, X[validation], Y[validation])

                if(performance[1] > best_performance):
                    best_performance = performance[1]
                    best_model = model

    #Save the trained model
    model.save('best_model_Q3.h5')
    test_performance = evaluate_architecture(best_model, X_test, Y_test)
    predict_hidden(X_test)
    return best_model

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


if __name__ == "__main__":
    main()
