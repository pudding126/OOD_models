from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAvgPool2D, Input, BatchNormalization, Activation, Add
import tensorflow as tf
from tensorflow.keras.regularizers import l2

WEIGHT_DECAY = 0.0005
# 结构块
def block(x, filters, strides=2, conv_short=True):
    if conv_short:
        short_cut = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='valid')(x)
        short_cut = BatchNormalization(epsilon=1.001e-5)(short_cut)
    else:
        short_cut = x

    # 2层卷积
    x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)

    x = Add()([x, short_cut])
    x = Activation('relu')(x)

    return x


def Resnet34(inputs, classes):
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = block(x, filters=64, strides=1, conv_short=False)
    x = block(x, filters=64, strides=1, conv_short=False)
    x = block(x, filters=64, strides=1, conv_short=False)

    x = block(x, filters=128, strides=2, conv_short=True)
    x = block(x, filters=128, strides=1, conv_short=False)
    x = block(x, filters=128, strides=1, conv_short=False)
    x = block(x, filters=128, strides=1, conv_short=False)

    x = block(x, filters=256, strides=2, conv_short=True)
    x = block(x, filters=256, strides=1, conv_short=False)
    x = block(x, filters=256, strides=1, conv_short=False)
    x = block(x, filters=256, strides=1, conv_short=False)
    x = block(x, filters=256, strides=1, conv_short=False)
    x = block(x, filters=256, strides=1, conv_short=False)

    x = block(x, filters=512, strides=2, conv_short=True)
    x = block(x, filters=512, strides=1, conv_short=False)
    x = block(x, filters=512, strides=1, conv_short=False)

    x = GlobalAvgPool2D()(x)

    # Define the ODIN as specified in Section 3.1.1 of
    # https://arxiv.org/abs/2002.11297
    h = Dense(classes, kernel_initializer="he_normal")(x)

    g = Dense(1, kernel_regularizer=l2(WEIGHT_DECAY))(x)
    g = BatchNormalization()(g)
    g = Activation("sigmoid")(g)
    x = tf.math.divide(h, g)

    return x
