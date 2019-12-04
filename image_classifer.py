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
resize()
print(path+dirs[0]+new_dirs[0])

def resize():
    cnt=0;
    for i in range(len(val_dirs)):
      for j in range(len(val_new_dirs)):
        if os.path.isfile(val_path+val_dirs[i]+"/"+val_new_dirs[j]):
            im = Image.open(val_path+val_dirs[i]+"/"+val_new_dirs[j])
            f, e = os.path.splitext(val_path+val_dirs[i]+"/"+val_new_dirs[j])
            imResize = im.resize((200,200), Image.ANTIALIAS)
            imResize.save(f+".jpg", 'JPEG', quality=90)
            print(val_path+val_dirs[i]+"/"+val_new_dirs[j])
    print(val_path+val_dirs[0]+"/"+val_new_dirs[0])
    
resize()

im = Image.open('/content/drive/My Drive/family photos/train/Srivalli/IMG_20191006_225118')
width, height = im.size
print(width,height)

batch_size = 20
IMG_SHAPE = 200 


image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE),
                                                class_mode='sparse'
                                                )
                                                
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')
                                                 
model = Sequential()

model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 80

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()