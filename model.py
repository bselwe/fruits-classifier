from keras.layers import Flatten, Dense
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers import average

kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 128 # the FC layer will have 128 neurons
l2_lambda = 0.0001 # use 0.0001 as a L2-regularisation factor
ens_models = 3 # to train three separate models on the data

class ModelBuilder(object):
    def build(self, input_shape, num_classes):
        inp = Input(shape=input_shape)
        inp_norm = BatchNormalization()(inp)
        outs = [] # the list of ensemble outputs

        for i in range(ens_models):
            # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
            conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(inp_norm)
            conv_1 = BatchNormalization()(conv_1)

            conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(conv_1)
            conv_2 = BatchNormalization()(conv_2)

            pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
            drop_1 = Dropout(drop_prob_1)(pool_1)

            # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
            conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(drop_1)
            conv_3 = BatchNormalization()(conv_3)

            conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(conv_3)
            conv_4 = BatchNormalization()(conv_4)

            pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
            drop_2 = Dropout(drop_prob_1)(pool_2)
            
            # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
            flat = Flatten()(drop_2)

            hidden = Dense(hidden_size, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_lambda))(flat)
            hidden = BatchNormalization()(hidden)

            drop_3 = Dropout(drop_prob_2)(hidden)
            outs.append(Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(l2_lambda), activation='softmax')(drop_3)) 
        
        out = average(outs)
        return Model(inputs=inp, outputs=out)