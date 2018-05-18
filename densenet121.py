# -*- coding: utf-8 -*-

# Courtesy of https://github.com/flyyufelix/cnn_finetune

from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from sklearn.metrics import log_loss
from sklearn.utils import class_weight
from custom_layers.scale_layer import Scale
from load_data import load_data

import time
import logging
import math


NUM_CLASSES = 18
NUM_EPOCH = 10

def densenet121_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    DenseNet 121 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121


    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7,7), strides=(2,2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/densenet121_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)


    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1,1), name=conv_name_base+'_x1', use_bias=False)(x)


    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3,3), name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1,1), name=conv_name_base, use_bias=False)(x)


    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


if __name__ == '__main__':

    train_dir = 'train_dir/'
    img_rows, img_cols = 64, 64 # Resolution of inputs
    channel = 3
    num_classes = NUM_CLASSES
    batch_size = 256
    nb_epoch = NUM_EPOCH
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = load_data(train_dir, img_rows, img_cols, num_classes)

    # Save_weights_only:
    #   - True: save model architecture separate from weights.
    #            Model architecture saved in model_architectures/, weights in saved_weights/
    #   - False: save entire model (architecture + weights) into folder 'saved_models/',
    save_weights_only = False

    # Create ModelCheckpoint to save model during training
    # If model validation acc improves after epoch, save at end of epoch
    # From https://machinelearningmastery.com/check-point-deep-learning-models-keras/
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    if save_weights_only:
        filepath="saved_weights/"+curr_time+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     save_weights_only=True)
    else:
        filepath="saved_models/"+curr_time+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     save_weights_only=False)

    callbacks_list = [checkpoint]

    # Load our model
    model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

    if save_weights_only:
        # Save the model architecture
        with open('model_architectures/model_architecture.json', 'w') as f:
            f.write(model.to_json())

    # Augment data with flips horizontally + vertically, and normalization
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='constant',
        cval=0
    )

    datagen.fit(X_train)

    TOTAL_IMGS = math.floor(len(X_train)*1.5)

    # Compute balanced class weights
    # Convert labels to integer format for sklearn's compute_class_weight
    y_ints = [y.argmax() for y in Y_train]
    class_weight = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_ints),
                                                     y_ints)
    class_weight_dict = dict(enumerate(class_weight))


    start = time.time()
    # Start Fine-tuning
    model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                                     #save_to_dir='augmented_imgs/norm_horz_vert_rot_shift_0'),
                        steps_per_epoch = TOTAL_IMGS,
                        epochs=nb_epoch,
                        shuffle=True,
                        verbose=1,
                        validation_data=(X_valid, Y_valid),
                        callbacks=callbacks_list,
                        class_weight=class_weight_dict)

    end = time.time()
    seconds_elapsed = end-start

    # Make predictions
    predictions_valid = model.predict(X_test, verbose=1)
    predictions_valid = np.argmax(predictions_valid, axis = 1)

    accuracy = np.sum(predictions_valid == Y_test)/len(Y_test)
    print("Accuracy: " + str(accuracy))

    f = open('log.txt', 'a+')
    f.write('\n' + curr_time + \
            '\n Mean: ' + str(datagen.mean) + \
            '\n Std: ' + str(datagen.std) + \
            '\n Num train images (including augmented): ' + str(TOTAL_IMGS) + \
            '\n Num epochs: ' + str(NUM_EPOCH) + \
            '\n Time elapsed (minutes): ' + ('%.3f' % (seconds_elapsed/60)) + \
            '\n Test accuracy: ' + str(accuracy))
    f.close()



