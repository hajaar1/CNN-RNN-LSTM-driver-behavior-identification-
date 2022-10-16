from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras import optimizers


!git clone --recursive https://github.com/titu1994/MLSTM-FCN.git
import sys
sys.path.insert(0, 'MLSTM-FCN/utils')
from layer_utils import AttentionLSTM

def generate_model():
    #ip = Input(shape=(valid_data.shape[1], valid_data.shape[2]))
    ip = Input(shape=(data.shape[1], data.shape[2]))
    x = Permute((2, 1))(ip)
    x=AttentionLSTM(10, return_sequences=True)(x)
    x=AttentionLSTM(10, return_sequences=True)(x)
    x=AttentionLSTM(10)(x)
    x = Dropout(0.8)(x)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])
    out = Dense(10, activation='softmax')(y)
    model = Model(ip, out)
    #model.summary()

    # add load model code here to fine-tune
    return model

model=generate_model()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=90, batch_size=128, verbose=2)
