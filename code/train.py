from keras.models import Model
from keras.utils import np_utils
import cifar_input
import keras
import json
import os

bs = [16,32,64,128]


if os.path.exists("model.h5"):
    model = keras.models.load_model("model.h5")
    data_train = cifar_input.cifar10()
    X_train, Y_train = data_train.cifar10All()
    Y_train = np_utils.to_categorical(Y_train, num_classes=10)
    data_test = cifar_input.cifar10('test')
    X_test, Y_test = data_test.cifar10All()
    Y_test = np_utils.to_categorical(Y_test, num_classes=10)
    with open("parameter.txt",'r') as file:
        best_run = json.loads(file.read())
    while True:
        model.fit(x=X_train, y=Y_train, batch_size=bs[best_run['batch_size']], epochs=30, 
                  validation_data = (X_test, Y_test),
                  callbacks=[keras.callbacks.EarlyStopping(monitor='acc'),
                  keras.callbacks.ModelCheckpoint("model.h5", period=5),
                  keras.callbacks.TensorBoard(write_grads=True, write_images=True)])
        
        