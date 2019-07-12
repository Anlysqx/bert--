import tensorflow as tf
import numpy as np
from bert import modeling
from bert import tokenization
from bert import optimization
import os
import pandas as pd

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_batch_size', 32, 'define the train batch size')
flags.DEFINE_integer('num_train_epochs', 3, 'define the num train epochs')
flags.DEFINE_float('warmup_proportion', 0.1, 'define the warmup proportion')
flags.DEFINE_float('learning_rate', 5e-5, 'the initial learning rate for adam')
flags.DEFINE_bool('is_traning', True, 'define weather fine-tune the bert model')

data = pd.read_csv('data/event_type_entity_extract_train.csv', encoding='UTF-8', header=None)
data = data[data[2] != u'其他']
classes = set(data[2])

train_data = []
for t, c, n in zip(data[1], data[2], data[3]):
    train_data.append((t.strip(), c.strip(), n.strip()))
np.random.shuffle(train_data)


def get_start_end_index(text, subtext):
    for i in range(len(text)):
        if text[i:i + len(subtext)] == subtext:
            return (i, i + len(subtext) - 1)
    return (-1, -1)


tmp_train_data = []
for item in train_data:
    start, end = get_start_end_index(item[0], item[2])
    if (start != -1) and (end != -1) and (end <= 480):
        tmp_train_data.append(item)

train_data = tmp_train_data
np.random.shuffle(train_data)

data = pd.read_csv('data/event_type_entity_extract_train.csv', encoding='UTF-8', header=None)
test_data = []
for t, c in zip(data[1], data[2]):
    test_data.append((t.strip(), c.strip()))

train_data = [str(item) for item in train_data]
# with open('data/train_data.txt',encoding='UTF-8',mode='a') as fp:
#     fp.write('\n'.join(train_data)+'\n')

for item in train_data:
    print(item)
