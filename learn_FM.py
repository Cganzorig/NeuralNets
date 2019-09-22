import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential, load_model, h5py
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
from sklearn.model_selection import KFold

from illustrate import illustrate_results_ROI

#######################################################################

def evaluate_architecture(model, X, Y):
    accuracy = model.evaluate(X, Y, verbose=1)
    return accuracy

#######################################################################

def predict_hidden(dataset):
    #Nomalize the data
    normalizer = preprocessing.Normalizer().fit(dataset)
    dataset = normalizer.transform(dataset)

    X = dataset[:,:3]
    Y = dataset[:,3:]

    model = load_model('best_model_Q2.h5')

    result = model.predict(X)
    print(result)
    return result

#######################################################################

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

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

                model.add(Dense(3, activation='softmax'))

                Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
                model.compile(Adam, loss='categorical_crossentropy', metrics=['accuracy'])
                model.fit(X[train], Y[train], batch_size=10, epochs=10, shuffle=True, verbose = 1)

                # # list all data in history
                # print(history.history.keys())
                # # summarize history for accuracy
                # plt.plot(history.history['acc'])
                # plt.title('model accuracy')
                # plt.ylabel('accuracy')
                # plt.xlabel('epoch')
                # plt.legend(['train', 'test'], loc='upper left')
                # plt.show()
                # # summarize history for loss
                # plt.plot(history.history['loss'])
                # plt.title('model loss')
                # plt.ylabel('loss')
                # plt.xlabel('epoch')
                # plt.legend(['train', 'test'], loc='upper left')
                # plt.show()

                performance = evaluate_architecture(model, X[validation], Y[validation])
                #performance = [loss, accuracy]

                if(performance[1] > best_performance):
                    best_performance = performance[1]
                    best_model = model

    #Save the trained model
    model.save('best_model_Q2.h5')
    test_performance = evaluate_architecture(best_model, X_test, Y_test)
    predict_hidden(X_test)
    return best_model

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


if __name__ == "__main__":
    main()
