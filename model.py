from keras import backend as K
from jugglingdataloader import JugglingDataLoader
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, LeakyReLU, BatchNormalization
from keras import regularizers
from keras.callbacks import ModelCheckpoint


def trainModel():
    augFactor = 1
    imageGenerator = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.1 * augFactor,
                                        height_shift_range=0.1 * augFactor,
                                        zoom_range=0.15 * augFactor)  # zoom_range was 0.15
    jugglingDataLoader = JugglingDataLoader(shape=(64, 64), gridShape=(15, 15), batch_size=8, expressFactor=1.0,
                                            imageGenerator=imageGenerator, dataType='SUBMOVAVG')

    valx, valy = jugglingDataLoader.getValidationSet()

    nRows, nCols, nDims = valx.shape[1:]
    w = 8
    l2_conv = 0  # 0.00000001

    model = Sequential()
    model.add(Conv2D(w, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_conv),
                     input_shape=(nRows, nCols, nDims)))
    model.add(LeakyReLU())
    model.add(Conv2D(w, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_conv)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())

    model.add(Conv2D(w * 2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_conv)))
    model.add(LeakyReLU())
    model.add(Conv2D(w * 2, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_conv)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())

    model.add(Conv2D(w * 4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_conv)))
    model.add(LeakyReLU())
    model.add(Conv2D(w * 4, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_conv)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D())

    l2_dense = 0  # 0.000001 * 5

    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(w * 64, kernel_regularizer=regularizers.l2(l2_dense)))
    model.add(LeakyReLU())
    model.add(Dense(15 * 15 * 9, activation='sigmoid'))
    model.add(Reshape((15, 15, 9)))

    model.summary()

    model.compile(optimizer='Adadelta', loss=grid_loss_with_hands)
    checkpoint = ModelCheckpoint('grid_model_submovavg_64x64_light.h5', verbose=1, save_best_only=True, period=1)

    model.fit_generator(
        jugglingDataLoader,
        workers=4,
        use_multiprocessing=True,
        epochs=300,
        callbacks=[checkpoint],
        validation_data=(valx, valy)
    )


def grid_loss(y_true, y_pred):
    true_boxes = y_true[:, :, :, 0]
    pred_boxes = y_pred[:, :, :, 0]
    box_loss = K.binary_crossentropy(true_boxes, pred_boxes)
    pos_xloss = true_boxes * K.binary_crossentropy(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    pos_yloss = true_boxes * K.binary_crossentropy(y_true[:, :, :, 2], y_pred[:, :, :, 2])
    return box_loss + pos_xloss + pos_yloss


def grid_loss_with_hands(y_true, y_pred):
    ball_loss = grid_loss(y_true[:, :, :, 0:3], y_pred[:, :, :, 0:3])
    rhand_loss = grid_loss(y_true[:, :, :, 3:6], y_pred[:, :, :, 3:6])
    lhand_loss = grid_loss(y_true[:, :, :, 6:9], y_pred[:, :, :, 6:9])
    return ball_loss + rhand_loss + lhand_loss


if __name__ == "__main__":
    trainModel()
