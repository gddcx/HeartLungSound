import random
import os
import shutil
import glob

PATH = '../data/app2temp/'
if not os.path.exists(PATH+'train'):
    os.makedirs(PATH+'train')
if not os.path.exists(PATH+'eval'):
    os.makedirs(PATH+'eval')
disease_id = {}
with open('../data/diagnosis.txt') as w:
    for line in w.readlines():
        patientid_disease = line.strip().split()
        id = patientid_disease[0]
        disease = patientid_disease[1]
        disease_id[id] = disease
disease_id_list = {'Healthy': [], 'URTI': [], 'COPD': [], 'Bronchiectasis': [], 'Bronchiolitis': [], 'Pneumonia': []}
ids = random.sample(range(101, 227), 126)
for i in ids:
    try:
        path = glob.glob(os.path.join(PATH, str(i)+'*.wav'))
        disease = disease_id[str(i)]
        disease_id_list[disease].append(path)
    except:
        pass
# 验证集
# print(disease_id_list)
for k, v in disease_id_list.items():
    count = len(v)//5
    data = random.sample(disease_id_list[k], count)
    for i in data:
        for j in i:
            shutil.move(j, PATH+'eval')
# 训练集
paths = glob.glob(os.path.join(PATH, '*.wav'))
for p in paths:
    shutil.move(p, PATH+'train')