import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, Conv2D, AveragePooling1D, MaxPooling2D
from keras.initializers import VarianceScaling
from keras.utils import np_utils

np.random.seed(123)

EPOCH = 10
TRAIN = './dataset/train.fasta'
TEST = './dataset/test_SP.fasta_out'
VALIDATION_RATE = 0.25
LENGTH = 108

# build CNN model
# 40 20 10 5
input_1 = Input(shape=(LENGTH, 20))

conv1d_1_bias_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
conv1d_1 = Conv1D(filters=40, kernel_size=5,
                  activation='relu', padding='same',
                  bias_initializer=conv1d_1_bias_init)(input_1)

avg_pool1d_1 = AveragePooling1D(pool_size=2, strides=2)(conv1d_1)

conv1d_2_bias_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
conv1d_2 = Conv1D(filters=20, kernel_size=5,
                  activation='relu', padding='same',
                  bias_initializer=conv1d_2_bias_init)(avg_pool1d_1)

avg_pool1d_2 = AveragePooling1D(pool_size=2, strides=2)(conv1d_2)

conv1d_3_bias_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
conv1d_3 = Conv1D(filters=10, kernel_size=5,
                  activation='relu', padding='same',
                  bias_initializer=conv1d_3_bias_init)(avg_pool1d_2)

avg_pool1d_3 = AveragePooling1D(pool_size=2, strides=2)(conv1d_3)

flatten_1 = Flatten()(avg_pool1d_3)

dense_1_bias_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
dense_1 = Dense(16, activation='relu', bias_initializer=dense_1_bias_init)(flatten_1)

dropout_1 = Dropout(0.25)(dense_1)

dense_2_bias_init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
output_1 = Dense(2, activation='softmax', bias_initializer=dense_2_bias_init)(dropout_1)

model = Model(inputs=input_1, outputs=output_1)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from helper import readdata
X, Y = readdata(TRAIN, LENGTH, 'train')

model.fit(X, Y,
         batch_size=32, epochs=EPOCH, verbose=1, validation_split=VALIDATION_RATE)

#model.fit(X, Y,
#          epochs=EPOCH, verbose=1, validation_split=VALIDATION_RATE)

model.save_weights('model_weights.h5')

# In prediction, index 0 is for signal peptide, index 1 for other.
"""
X_test = readdata(TEST, LENGTH, 'test')
Y_pred = model.predict(X_test)
pred = np.argmax(Y_pred, axis=1)
"""


# print prediction result of positive sample
"""
print(np.sum((pred == 0).astype(np.int8)))
print(pred.shape[0])
print(np.sum((pred == 0).astype(np.int8))/pred.shape[0])
"""


with open('./json_model.json', 'w') as f:
    f.write(model.to_json())
