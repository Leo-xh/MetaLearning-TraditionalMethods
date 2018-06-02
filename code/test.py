from keras import applications
from keras.utils import np_utils
import cifar_input
import keras
nas = applications.nasnet
NASNet = nas.NASNet(input_shape=(32,32,3), penultimate_filters=48, num_blocks=2, 
                    stem_block_filters=2, classes=10)
data_train = cifar_input.cifar10()
X_train, Y_train = data_train.cifar10All()
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
data_test = cifar_input.cifar10('test')
X_test, Y_test = data_test.cifar10All()
Y_test = np_utils.to_categorical(Y_test, num_classes=10)

NASNet.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
NASNet.fit(x=X_train, y=Y_train, batch_size=32, epochs=30,
                  validation_data=(X_test, Y_test),
                  callbacks=[keras.callbacks.EarlyStopping(monitor='acc'),
                             keras.callbacks.ModelCheckpoint(
                                 "model.h5", period=5),
                             keras.callbacks.TensorBoard(write_grads=True, write_images=True)])