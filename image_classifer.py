import tensorflow as tf
import os as os
from PIL import Image
import sys
import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.chdir("drive/My Drive/photos");

base_dir = os.getcwd()

classnames=["abc ","def","ghi","jkl","mnop"];

for cl in classnames:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  num_train = int(round(len(images)*0.8))
  train, val = images[:num_train], images[num_train:]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))
    
round(len(images)*0.8)


train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
print(train_dir)
print(val_dir)

path="/content/drive/My Drive/photos/train/"
dirs =os.listdir( path )
new_dirs=[];
for i in range(len(dirs)):
  new_dirs.append(os.listdir( path+"/"+dirs[i] ));
new_dirs=list(np.concatenate((new_dirs), axis=None))
# print(type(new_dirs))
# print(type(dirs))
# dirs

val_path="/content/drive/My Drive/photos/val/"
val_dirs =os.listdir( val_path )
val_new_dirs=[];
for i in range(len(val_dirs)):
  val_new_dirs.append(os.listdir( val_path+"/"+val_dirs[i] ));
val_new_dirs=list(np.concatenate((val_new_dirs), axis=None))
# print(val_dirs);
# val_new_dirs

def resize():
    cnt=0;
    for i in range(len(dirs)):
      for j in range(len(new_dirs)):
        if os.path.isfile(path+dirs[i]+"/"+new_dirs[j]):
            im = Image.open(path+dirs[i]+"/"+new_dirs[j])
            f, e = os.path.splitext(path+dirs[i]+"/"+new_dirs[j])
            imResize = im.resize((200,200), Image.ANTIALIAS)
            imResize.save(f+".jpg", 'JPEG', quality=90)
            print(path+dirs[i]+"/"+new_dirs[j])
        else:
          cnt+=1
    print(cnt)
