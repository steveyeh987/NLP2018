import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
import gensim
import re
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.utils import plot_model

def entity_extraction(lines):
    """
    remove puntuations in sentences and extract entities
    """
    e1 = re.search('<e1>(.+?)</e1>', lines)
    e2 = re.search('<e2>(.+?)</e2>', lines)
    lines = lines.strip("\n").split("\t", 1)[-1]
    lines = re.sub(e1.group(0), '_'.join(e1.group(1).split(" ")), lines)
    lines = re.sub(e2.group(0), '_'.join(e2.group(1).split(" ")), lines)
    lines = re.sub("'s", '', lines)
    lines = re.sub(r'|'.join(map(re.escape, ['\'', '"', '!', ',', '?', ':', '.', ';', '(', ')'])), '', lines)
    lines = lines.split(" ")
    return lines, '_'.join(e1.group(1).split(" ")), '_'.join(e2.group(1).split(" "))

def preprocess(entity):
    """
    get corresponding entity vector in the trained word2vec model 
    """
    X_train_input1 = []
    X_train_input2 = []

    for pair in entity:
        e1, e2 = pair[0], pair[1]
        X_train_input1.append(w2v_model.wv[e1])
        X_train_input2.append(w2v_model.wv[e2])
    X_train_input1 = np.array(X_train_input1)
    X_train_input2 = np.array(X_train_input2)
    return X_train_input1, X_train_input2

sentences = []
x_train = []
y_train = []
x_test = []

with open("TRAIN_FILE.txt", "r") as file:
    for i, lines in enumerate(file):
        if (i+1) % 4 == 1:
            lines, e1, e2 = entity_extraction(lines)
            sentences.append(lines)
        elif (i+1) % 4 == 2:
            lines = lines.strip("\n")
            entity = [e1, e2]
            x_train.append(entity)
            y_train.append(lines)
            
        else:
            continue

with open("TEST_FILE.txt", "r") as file:
    for lines in file:
        lines, e1, e2 = entity_extraction(lines)
        sentences.append(lines)
        lines = [e1, e2]
        x_test.append(lines)

w2v_model = gensim.models.Word2Vec(sentences, size=128, window=3, min_count=0, sg=1, workers=20, iter=100)

X_train_input1, X_train_input2 = preprocess(x_train)
X_test_input1, X_test_input2 = preprocess(x_test)

# generate dictionaries

Relations = list(set(y_train))
r_to_id = {}
id_to_r = {}
Y_train = np.zeros((len(y_train), len(Relations)))

for i, relation in enumerate(Relations):
    r_to_id[relation] = i
    id_to_r[i] = relation

for i, relation in enumerate(y_train):
    Y_train[i, r_to_id[relation]] = 1

# preparing classifier model

input1 = Input(shape=(128,), dtype='float32')
fc1 = Dense(64, activation='relu')(input1)
input2 = Input(shape=(128,), dtype='float32')
fc2 = Dense(64, activation='relu')(input2)
fc = concatenate([fc1, fc2])
fc = Dense(64, activation='relu')(fc)
fc = Dense(32, activation='relu')(fc)
output = Dense(19, activation='softmax')(fc)

model = Model(inputs=[input1, input2], outputs=[output])
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([X_train_input1, X_train_input2], [Y_train],
          epochs=10, batch_size=64)

# prediction and wirte file

predict = model.predict([X_test_input1, X_test_input2])
predict = np.argmax(predict, 1)

relation = []

for p in predict:
    relation.append(id_to_r[p])

with open("predict.txt", "w") as f:
    for i, r in enumerate(relation):
        f.write("{}\t{}\n".format(str(i+8001), r))

plot_model(model, to_file='model.png')

