from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Activation, BatchNormalization
from keras.layers import Softmax, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
import keras
import os
import cifar_input

def Cifarmodel():
    In = Input(shape=(32,32,3))
    X = BatchNormalization()(In)
    X = Conv2D(32,(3,3))(X)
    X = MaxPooling2D()(X)
    X = Activation('relu')(X)
    X = Conv2D(64,(4,4))(X)
    X = Activation('relu')(X)
    X = AveragePooling2D()(X)
    X = Conv2D(64,(5,5))(X)
    X = Activation('relu')(X)
    X = AveragePooling2D()(X)
    X = BatchNormalization()(X)
    X = Flatten()(X)
    X = Dense(10, activation='relu')(X)
    X = Activation('softmax')(X)
    return Model(inputs=In, outputs=X)

data_train = cifar_input.cifar10()
X_train, Y_train = data_train.cifar10All()
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
data_test = cifar_input.cifar10('test')
X_test, Y_test = data_test.cifar10All()
Y_test = np_utils.to_categorical(Y_test, num_classes=10)
lr = 1e-4
batch_size = 32
model = Cifarmodel()
keras.utils.print_summary(model)
keras.utils.plot_model(model,show_shapes=True, show_layer_names=False)
adam = Adam(lr=lr)
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
if(os.path.exists("model.h5")):
    model.load_weights("model.h5")
his = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=50, callbacks=
                [keras.callbacks.ModelCheckpoint("model.h5", period=5),
                 keras.callbacks.EarlyStopping(monitor='acc'),
                 keras.callbacks.TensorBoard()]
                )
loss, accuracy = model.evaluate(x=X_test, y=Y_test, batch_size=batch_size)
print("accuracy = ", accuracy)
model.save("model.h5")