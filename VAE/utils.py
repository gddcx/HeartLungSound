import glob
from PIL import Image

# 统计各类别的数量
def statistic_files_class():
    PATH = '../../data/ICBHI_final_database/*.wav'
    files = glob.glob(PATH)

    patients_disease_dict = {}
    with open('../../data/diagnosis.txt') as f:
        for l in f.readlines():
            lines_list = l.strip().split()
            patients_disease_dict[lines_list[0]] = lines_list[1]

    CLASS='COPD'
    one_class_list = []
    disease_index = {'Healthy': 0, 'URTI': 0, 'LRTI': 0, 'Asthma': 0, 'COPD': 0,
                          'Bronchiectasis': 0, 'Bronchiolitis': 0, 'Pneumonia': 0}
    for p in files:
        id = p.split('/')[-1].split('_')[0]
        # 统计某一类的ID
        if patients_disease_dict[id]==CLASS:
            one_class_list.append(id)
        disease_index[patients_disease_dict[id]]+=1
    one_class_list.sort()
    one_class_set = set(one_class_list)
    print(CLASS, one_class_set)
    print('ALL CLASS ', disease_index)

# 计算平均列数
def statistic_columns():
    PATH = '../../data/VAE_128/*png'
    files = glob.glob(PATH)
    total_cols = 0
    for f in files:
        img = Image.open(f)
        img_size = img.size
        cols = img_size[0]
        total_cols += cols
    print(total_cols / len(files))

# 分割数据集
def split_dataset(root_path, label_file):
    import os
    import random
    import shutil
    train_path = os.path.join(root_path, 'train')
    eval_path = os.path.join(root_path, 'eval')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    dict_id_disease = {}
    dict_disease_idlist = {'URTI':[], 'Healthy': [], 'COPD': [], 'Pneumonia': [], 'Bronchiolitis': [], 'Bronchiectasis': []}
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            id, disease = l.split()
            id = id.strip()
            disease = disease.strip()
            try:
                dict_disease_idlist[disease].append(id)
                dict_id_disease[id] = disease
            except Exception as e:
                print(e)
    eval_dict = {}
    for k, v in dict_disease_idlist.items():
        samples = random.sample(v, 5)
        # 部分COPD已经被删除了
        for s in samples:
            if not glob.glob(os.path.join(root_path, s + '*')):
                samples.remove(s)
        eval_dict[k] = samples[:3]
        print(eval_dict[k])
    spectrumgraphs = glob.glob(os.path.join(root_path, '*.png'))
    for stg in spectrumgraphs:
        id = os.path.basename(stg).split('_')[0]
        disease = dict_id_disease[id]
        # 如果id在验证集列表里
        if id in eval_dict[disease]:
            shutil.copy(stg, eval_path)
        else:
            shutil.copy(stg, train_path)


if __name__ == '__main__':
    statistic_columns()
    # 数量可能会少，因为COPD已经删掉了一部分，所以部分ID不存在
    # split_dataset('../../data/VAE_128', '../../data/diagnosis.txt')
