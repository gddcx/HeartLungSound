import glob
import scipy.io as scio
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

annotation_path = '../../data/diagnosis.txt'
disease_index = {'Healthy':0, 'URTI':1, 'COPD':2,
                     'Bronchiectasis':3, 'Bronchiolitis':4, 'Pneumonia':5}
patients_disease_dict = {}
with open(annotation_path) as f:
    for l in f.readlines():
        lines_list = l.strip().split()
        try:
            patients_disease_dict[lines_list[0]] = disease_index[lines_list[1]]
        except KeyError:
            print(lines_list[1])
l_x = []
l_y = []
data_path = '../../data/approch2/train/*.mat'
data_files = glob.glob(data_path)
for p in data_files:
    mat = scio.loadmat(p)
    mat = mat['c']
    mat = mat.transpose()
    mat = mat.reshape(1, -1)
    l_x.append(mat)
    patient_id = os.path.basename(p).split('_')[0]
    l_y.append(patients_disease_dict[patient_id])
x_train = np.concatenate(l_x, axis=0)
y_train = np.asarray(l_y)

l_x = []
l_y = []
data_path = '../../data/approch2/eval/*.mat'
data_files = glob.glob(data_path)
for p in data_files:
    mat = scio.loadmat(p)
    mat = mat['c']
    mat = mat.transpose()
    mat = mat.reshape(1, -1)
    l_x.append(mat)
    patient_id = os.path.basename(p).split('_')[0]
    l_y.append(patients_disease_dict[patient_id])
x_test = np.concatenate(l_x, axis=0)
y_test = np.asarray(l_y)

rf = RandomForestClassifier()
param = {"n_estimators":[30, 60, 120,200,300,500,800,1200],"max_depth":[5,8,15,25,30]}
gc = GridSearchCV(rf, param_grid=param, cv=2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

gc.fit(x_train, y_train)

print('Accuracy: ', gc.score(x_test, y_test))
best_rf = gc.best_estimator_
predict = best_rf.predict(x_test)
confusion_matrix = np.zeros((6,6),)
for p, t in zip(predict, l_y):
    confusion_matrix[p, t]+=1
recall = confusion_matrix.diagonal() / (confusion_matrix.sum(0)+1e-16)
precision = confusion_matrix.diagonal() / (confusion_matrix.sum(1)+1e-16)
F1_score = 2*recall*precision/(recall+precision+1e-16)
print(recall, precision, F1_score)