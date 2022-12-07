import os.path
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

train_size=0.8
path = 'H:/519/data/COVID-19_Radiography_Dataset'
csv_names = ['COVID.metadata.csv','VIRAL.metadata.csv','NORMAL.metadata.csv']
csv_name = csv_names[2] # choose class name
df = pd.read_csv(os.path.join(path,csv_name), low_memory=False)
X = df['FILE NAME']

# split size 0.8 0.1 0.1
X_train, X_rem = train_test_split(X, train_size=0.8)

test_size = 0.5 #split 20% to 0.1, 0.1
X_valid, X_test = train_test_split(X_rem, test_size=0.5)

print('Train shape:',X_train.shape,', Valid shape:', X_valid.shape, 'Test shape:', X_test.shape)

train_dir = './data/NORMAL/train/'
test_dir = './data/NORMAL/test/'
val_dir = './data/NORMAL/validation/'

os.mkdir('./data/NORMAL')
os.mkdir(train_dir)
os.mkdir(test_dir)
os.mkdir(val_dir)

# Providing the folder path
origin = os.path.join(path,'NORMAL\\images\\')
target = train_dir


for file_name in X_train:
   file_name += '.PNG'
   shutil.copy(origin+file_name, target+file_name)

target = val_dir
for file_name in X_valid:
   file_name += '.PNG'
   shutil.copy(origin+file_name, target+file_name)

target = test_dir
for file_name in X_test:
   file_name += '.PNG'
   shutil.copy(origin+file_name, target+file_name)
