from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Activation, BatchNormalization
from keras.layers import Softmax, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


import keras
import os
import cifar_input
import random




def Cifarmodel(X_train, Y_train, X_test, Y_test, epochs):
    model = keras.Sequential()
    if conditional({{choice(['BN1True', 'BN1False'])}}) == 'BN1True':
        model.add(BatchNormalization(input_shape=(32,32,3)))
    model.add(Conv2D(32, (3, 3), input_shape=(32,32,3)))
    model.add({{choice([MaxPooling2D(), AveragePooling2D()])}})
    model.add(Activation('relu'))
    if conditional({{choice(['BN2True', 'BN2False'])}}) == 'BN2True':
        model.add(BatchNormalization())
    model.add(Conv2D(64,(3, 3)))
    model.add({{choice([MaxPooling2D(), AveragePooling2D()])}})
    model.add(Activation('relu'))
    if conditional({{choice(['BN3True', 'BN3False'])}}) == 'BN3True':
        model.add(BatchNormalization())
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    model.add({{choice([MaxPooling2D(), AveragePooling2D()])}})
    if conditional({{choice(['BN3True', 'BN3False'])}}) == 'BN3True':
        model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Activation('softmax'))
#    keras.utils.print_summary(model)
#    keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)
    model.compile({{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='categorical_crossentropy', metrics=['accuracy'])
    his = model.fit(x=X_train, y=Y_train, batch_size={{choice([16, 32, 64, 128])}}, epochs=epochs, callbacks=[keras.callbacks.EarlyStopping(
                                                                                                          monitor='acc')]
                    )
    loss, accuracy = model.evaluate(x=X_test, y=Y_test)
    print("accuracy =", accuracy)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

data_train = cifar_input.cifar10()
X_train, Y_train = data_train.cifar10All()
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
data_test = cifar_input.cifar10('test')
X_test, Y_test = data_test.cifar10All()
Y_test = np_utils.to_categorical(Y_test, num_classes=10)


def data():
        data_train = cifar_input.cifar10()
        X_train, Y_train = data_train.cifar10All()
        Y_train = np_utils.to_categorical(Y_train, num_classes=10)
        data_test = cifar_input.cifar10('test')
        X_test, Y_test = data_test.cifar10All()
        Y_test = np_utils.to_categorical(Y_test, num_classes=10)
        epochs = 2
        return X_train, Y_train, X_test, Y_test, epochs


K = 3
Turns = 5
ep = 2
bs = [16,32,64,128]
    
if __name__ == "__main__":
    models = []
    for j in range(Turns):
        print(j, "turn")
        randoms = {}
        for i in range(2*K):
            print("Finding new model", i)
            best_run, best_model = optim.minimize(
                    model=Cifarmodel, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
            for k in range(j-1):
                best_model.fit(x=X_train, y=Y_train, batch_size=bs[best_run['batch_size']], epochs=ep, callbacks=[keras.callbacks.EarlyStopping(monitor='acc')])
            models.append((best_model, best_model.evaluate(X_test, Y_test)[1], best_run))
            
        for model, acc, run in models:
            randoms[random.betavariate(acc, 1-acc)] = (model, run)
        Kmodels = []
        for (rand, (model, run)) in sorted(randoms.items())[:K]:
            Kmodels.append((model, run))
        models = []
        for model, run in Kmodels:
            model.fit(x=X_train, y=Y_train, batch_size=bs[run['batch_size']], epochs=ep, callbacks=[keras.callbacks.EarlyStopping(monitor='acc')])
            models.append((model, model.evaluate(X_test, Y_test)[1], run)) 
        print("In this turn, the K models remained are:")
        for model, acc, run in models:
            print("Acc:", acc, "Parameters:", run)                                                                                            
    for model, acc, run in models:
            randoms[random.betavariate(acc, 1-acc)] = (model, run)
    acc, (best_model, best_run) = sorted(randoms.items())[0]
    
    
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save("model.h5")
    with open("parameter.txt",'w') as config:
        config.write(str(best_run))
    
