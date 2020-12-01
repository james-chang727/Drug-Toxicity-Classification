import tensorflow as tf
import pickle
import numpy as np 
from tensorflow import keras

score_data = np.array(pickle.load(open('../SR-ARE-score/names_onehots.pickle', 'rb'))['onehots']).reshape(-1,70,325,1)

model = keras.models.load_model('saved_models/CNN_model4.h5', compile=False)
predictions = model.predict(score_data.astype(np.float32))

stk = []
for i in predictions: 
    for j in i:
        if float(j) < 0.0005: output = 0
        else: output = 1
    stk.append(output)

with open('labels.txt', 'w') as f:
    for item in stk:
        f.write(str(item)+'\n')
f.close() 