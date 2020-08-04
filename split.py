from pydub import AudioSegment
import glob
import os
import random
import shutil
import numpy as np
import librosa


# 数据增强的方式
# 滚动
def roll(path, times):
    file_name = os.path.basename(path).split('.')[0]
    dir_name = os.path.dirname(path)
    y, sr = librosa.load(path, sr=None)
    y = np.roll(y, sr//10)
    librosa.output.write_wav(os.path.join('../data/data_augmentation', str(file_name)+'_roll_'+str(times)+'.wav'), y, sr)
# 改变音高
def pitch_shifting(path, times):
    file_name = os.path.basename(path).split('.')[0]
    dir_name = os.path.dirname(path)
    y, sr = librosa.load(path, sr=None)
    rd = random.uniform(0.01, 2.0)
    y = librosa.effects.pitch_shift(y, sr, n_steps=rd)
    librosa.output.write_wav(os.path.join('../data/data_augmentation', str(file_name)+'_pitch_'+str(times)+'.wav'), y, sr)
#改变音速
def time_stretching(path, times):
    file_name = os.path.basename(path).split('.')[0]
    dir_name = os.path.dirname(path)
    y, sr = librosa.load(path, sr=None)
    rd = random.uniform(0.8, 1.2)
    y = librosa.effects.time_stretch(y, rd)
    librosa.output.write_wav(os.path.join('../data/data_augmentation', str(file_name)+'_stretch_'+str(times)+'.wav'), y, sr)
# 白噪声
def white_noise(path, times):
    file_name = os.path.basename(path).split('.')[0]
    dir_name = os.path.dirname(path)
    y, sr = librosa.load(path, sr=None)
    noise = 0.005*np.random.normal(0, 1, len(y))
    y = y + noise
    librosa.output.write_wav(os.path.join('../data/data_augmentation', str(file_name)+'_noise_'+str(times)+'.wav'), y, sr)
func_augmentation = [roll, pitch_shifting, time_stretching, white_noise]

# key 病人id , value 疾病
disease_id = {'Healthy': [], 'URTI': [], 'COPD': [],
                         'Bronchiectasis': [], 'Bronchiolitis': [], 'Pneumonia': []}
with open('../data/diagnosis.txt') as f:
    for l in f.readlines():
        line_list = l.strip().split()
        if line_list[1] in disease_id:
            disease_id[line_list[1]].append(line_list[0])
# 先去掉部分COPD
# direction = ['Tc', 'Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr']
# for id in disease_id['COPD']:
#     direct = random.sample(direction, 3)
#     p0 = glob.glob('/home/dengchangxing/HeartLungSound/data/ICBHI_final_database/' + id + '*' + direct[0]+'*')
#     p1 = glob.glob('/home/dengchangxing/HeartLungSound/data/ICBHI_final_database/' + id + '*' + direct[1]+'*')
#     p2 = glob.glob('/home/dengchangxing/HeartLungSound/data/ICBHI_final_database/' + id + '*' + direct[2]+'*')
#     print(p0)
#     try:
#         shutil.move(p0[0].split('.')[0]+'.wav', '../data/deleteCOPD')
#         shutil.move(p0[0].split('.')[0]+'.txt', '../data/deleteCOPD')
#         shutil.move(p1[0].split('.')[0]+'.wav', '../data/deleteCOPD')
#         shutil.move(p1[0].split('.')[0]+'.txt', '../data/deleteCOPD')
#         shutil.move(p2[0].split('.')[0]+'.wav', '../data/deleteCOPD')
#         shutil.move(p2[0].split('.')[0]+'.txt', '../data/deleteCOPD')
#     except:
#         pass

# 按呼吸周期分割
# PATH='../data/ICBHI_final_database/'
# SAVE_PATH = '../data/split/'
# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)
# audio_paths = glob.glob(PATH+'*.wav')
# annotation_paths = [x.replace('.wav', '.txt') for x in audio_paths]
# for audio, annotation in zip(audio_paths, annotation_paths):
#     input = AudioSegment.from_wav(audio)
#     with open(annotation) as f:
#         file_name = annotation.split('/')[-1]
#         for i, l in enumerate(f.readlines()):
#             new_file_name = file_name.split('.')[0] + '_' + str(i) + '.wav'
#             line_list = l.strip().split()
#             # 转为ms
#             start = float(line_list[0])*1000
#             end = float(line_list[1])*1000
#             if audio == '../data/ICBHI_final_database/120_1b1_Al_sc_Meditron.wav':
#                 print(len(audio), line_list, annotation)
#             part = input[start:end]
#             part.export(SAVE_PATH + new_file_name, format='wav')

# # 数据增强
if not os.path.exists('../data/data_augmentation'):
    os.makedirs('../data/data_augmentation')
augmentation_time = {'Healthy': 160, 'URTI': 170,
                         'Bronchiectasis': 180, 'Bronchiolitis': 183, 'Pneumonia': 160}
for k, v in augmentation_time.items():
    # 对应疾病的所有病人ID
    id_list = disease_id[k]
    # 数据增强次数
    for i in range(v):
        # 随机选一个病人
        patient_id = random.choice(id_list)
        wav_list = glob.glob('../data/ICBHI_final_database/'+patient_id+'*.wav')
        print(wav_list)
        # 随机选一个病人的一条记录
        wav = random.choice(wav_list)
        # 随机一种增强方式
        func = random.choice(func_augmentation)
        try:
            func(wav, i)
        except:
            pass