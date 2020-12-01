import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = InteractiveSession(config=config)

"""Load the data and reshape to feed into CNN"""

train_data = np.array(pickle.load(open('./data/SR-ARE-train/names_onehots.pickle', 'rb'))['onehots']).reshape(-1,70,325,1)
test_data = np.array(pickle.load(open('./data/SR-ARE-test/names_onehots.pickle', 'rb'))['onehots']).reshape(-1,70,325,1)
# print('Training data shape:', train_data.shape)
# print('Testing data shape:', test_data.shape)

"""============================================================================="""

"""Get labels of each corresponding training and testing one hot vectors"""

def get_labels(file_path):
    with open(file_path, 'r') as f:
        labels = [int(line.rstrip().split(',')[-1]) for line in f]
        f.close()
    return labels

train_labels = np.array(get_labels('./data/SR-ARE-train/names_labels.txt'))
test_labels = np.array(get_labels('./data/SR-ARE-test/names_labels.txt'))
# print('Training labels shape:', train_labels.shape)
# print('Testing labels shape:', test_labels.shape)

"""=============================================================================="""

"""Model Design compilation and training"""

METRICS = [keras.metrics.TruePositives(name='tp'),
           keras.metrics.FalsePositives(name='fp'),
           keras.metrics.TrueNegatives(name='tn'),
           keras.metrics.FalseNegatives(name='fn'),
           keras.metrics.BinaryAccuracy(name='accuracy'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall')]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)

def CNN(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.models.Sequential([
        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=train_data.shape[1:]),
        keras.layers.MaxPooling2D((2,2)),
        # keras.layers.Dense(32, input_shape=train_data.shape[1:], activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='relu'),
        # keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',#keras.optimizers.RMSprop(lr=0.001),
                  metrics=metrics)
    return model

"""=============================================================================="""

"""Classify pos and neg data to deal with imbalance problems"""

pos_data = []
neg_data = []

for i in range(len(train_data)):
    label = train_labels[i]
    data = train_data[i]
    if label == 0:
        neg_data.append(data)
    else:
        pos_data.append(data)

pos_labels = [1 for i in range(len(pos_data))]
neg_labels = [0 for i in range(len(neg_data))]

"""=============================================================================="""

"""Oversample toxic labels to balance dataset"""

ids = np.arange(len(pos_data))
choices = list(np.random.choice(ids, len(neg_data)))

resampled_pos_data = np.array([pos_data[i] for i in choices])
resampled_pos_labels = np.array([pos_labels[i] for i in choices])

"""=============================================================================="""

"""Split training data further into validation and train datasets and make sure 
   train and val are both balanced in terms of toxic and non-toxic samples"""

def train_val_split(features, label_class, VAL_SAMPLES=2000):
    all_ids = np.arange(len(features))
    
    val_ids = list(np.random.choice(all_ids, VAL_SAMPLES, replace=False))
    val_data = np.array([features[i] for i in val_ids])
    val_labels = [1 if label_class==1 else 0 for i in range(len(val_ids))]

    train_ids = list(set(all_ids)-set(val_ids))
    train_data = np.array([features[i] for i in train_ids])
    train_labels = [1 if label_class==1 else 0 for i in range(len(train_ids))]

    return [train_data, np.array(train_labels), val_data, np.array(val_labels)]

pos_TV_split = train_val_split(resampled_pos_data, 1)
neg_TV_split = train_val_split(neg_data, 0)

train_data = np.concatenate([pos_TV_split[0], neg_TV_split[0]], axis=0)
train_labels = np.concatenate([pos_TV_split[1], neg_TV_split[1]], axis=0)
val_data = np.concatenate([pos_TV_split[2], neg_TV_split[2]], axis=0)
val_labels = np.concatenate([pos_TV_split[3], neg_TV_split[3]], axis=0)

def shuffle(X, Y):
    order = np.arange(len(Y))
    np.random.shuffle(order)
    X = X[order]
    Y = Y[order]

shuffle(train_data, train_labels)
shuffle(val_data, val_labels)

"""=============================================================================="""

"""Train and test accuracy of model, then save the model"""

model = CNN(output_bias=np.log(4855/4855))
model_history = model.fit(train_data, train_labels, batch_size=128, epochs=8, validation_data=(val_data, val_labels), verbose=2)

results = model.evaluate(test_data, test_labels, verbose=2)
print("Test Loss, Test Accuracy:", [results[0], results[-3]])

model.save('1155106843/saved_models/CNN_model5.h5')
