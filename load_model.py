from keras.models import load_model
from custom_layers.scale_layer import Scale
from custom_layers.scale_layer import Scale
from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
import numpy as np



MODEL_NAME = 'saved_models/model2.hdf5'

model = load_model(MODEL_NAME, custom_objects = {'Scale': Scale})

print(model.summary())

# Below code as suggestion on thread https://github.com/keras-team/keras/issues/7085
# to prevent batch normalization during test time
# Freeze batch norm
for layer in model.layers:
    layer.trainable = False
    if isinstance(layer, BatchNormalization):
        layer._per_input_updates = {}


#model.save('saved_models/model2_frozen_bn.hdf5')
