import random
import os
import shutil
import glob

PATH = '../data/split/'
if not os.path.exists('../data/train'):
    os.makedirs('../data/train')
if not os.path.exists('../data/eval'):
    os.makedirs('../data/eval')

paths_list = []
ids = random.sample(range(101, 227), 126)
for i in ids:
    path = glob.glob(os.path.join(PATH, str(i)+'*'))
    paths_list.append(path)
# 测试集
eval_list = paths_list[:20]
# 训练集
train_list = paths_list[20:]

for i in eval_list:
    for j in i:
        shutil.copy(j, '../data/eval')
for i in train_list:
    for j in i:
        shutil.copy(j, '../data/train')