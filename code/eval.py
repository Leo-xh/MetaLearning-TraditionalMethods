<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 19:39:17 2018

@author: lenovo
"""

=======
from keras.models import Model
from keras.utils import np_utils
import cifar_input
import keras
import json
import os

model = keras.models.load_model("model.h5")
data_test = cifar_input.cifar10('test')
X_test, Y_test = data_test.cifar10All()
Y_test = np_utils.to_categorical(Y_test, num_classes=10)
with open("parameter.txt",'r') as file:
    best_run = json.loads(file.read())
print(model.evaluate(X_test, Y_test))
    
    
>>>>>>> 8b093e5da4c8c7372d3325f5b14252b20a5abf55
