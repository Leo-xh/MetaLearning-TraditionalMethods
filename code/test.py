# In = Input(shape=(32, 32, 3))
    # X = In
    # if conditional({{choice(['BN1True', 'BN1False'])}}) == 'BN1True':
    #     X = BatchNormalization()(In)
    # X = Conv2D({{choice([32, 64, 128])}},
    #            ({{choice([3, 4, 5])}}, {{choice([3, 4, 5])}}))(X)
    # X = {{choice([MaxPooling2D()(X), AveragePooling2D()(X)])}}
    # X = Activation('relu')(X)
    # if conditional({{choice(['BN2True', 'BN2False'])}}) == 'BN2True':
    #     X = BatchNormalization()(X)
    # X = Conv2D({{choice([32, 64, 128])}},
    #            ({{choice([3, 4, 5])}}, {{choice([3, 4, 5])}}))(X)
    # X = {{choice([MaxPooling2D()(X), AveragePooling2D()(X)])}}
    # X = Activation('relu')(X)
    # if conditional({{choice(['BN3True', 'BN3False'])}}) == 'BN3True':
    #     X = BatchNormalization()(X)
    # X = Conv2D({{choice([32, 64, 128])}},
    #            ({{choice([3, 4, 5])}}, {{choice([3, 4, 5])}}))(X)
    # X = Activation('relu')(X)
    # X = {{choice([MaxPooling2D()(X), AveragePooling2D()(X)])}}
    # if conditional({{choice(['BN4True', 'BN4False'])}}) == 'BN4True':
    #     X = BatchNormalization()(X)
    # X = Flatten()(X)
    # X = Dense(10, activation='relu')(X)
    # X = Activation('softmax')(X)
    #    if(os.path.exists("model.h5")):
#        model.load_weights("model.h5")
#    keras.callbacks.ModelCheckpoint("model.h5", period=5),
    # model = Model(inputs=In, outputs=X)