import os
import cv2
import pandas
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import img_to_array
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization

from keras.layers import Dropout
from keras.layers import Input

from imblearn.over_sampling import SMOTE

from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

path = './samples/'
def t_img (img) :
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

def c_img (img) :
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,2), np.uint8))

def d_img (img) :
    return cv2.dilate(img, np.ones((2,2), np.uint8), iterations = 1)

def b_img (img) :
    return cv2.GaussianBlur(img, (1,1), 0)

def conv_layer (filterx) :
    
    model = Sequential()
    
    model.add(Conv2D(filterx, (3,3), padding = 'same', activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2)) #previene el sobre apendizaje
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
    
    return model

def dens_layer (hiddenx) :
    
    model = Sequential()
    
    model.add(Dense(hiddenx, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    return model

def cnn (filter1, filter2, filter3, hidden1, hidden2) :
    
    model = Sequential()
    model.add(Input((40, 20, 1,)))
    
    model.add(conv_layer(filter1))
    model.add(conv_layer(filter2))
    model.add(conv_layer(filter3))
    
    model.add(Flatten())
    model.add(dens_layer(hidden1))
    model.add(dens_layer(hidden2))
    
    model.add(Dense(19, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model
#inicializamos las entradas y salidas
X = []
y = []


#se lee todos los samples de captcha
for image in os.listdir(path) :
    
    if image[6:] != 'png' :
        continue
    
    #se le aplica toda la transformacion de imagen para que sea legible
    img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
    img = t_img(img)
    img = c_img(img)
    img = d_img(img)
    img = b_img(img)
    #se crea la cuadricula para detectar cdada cuadrante de cada imagen
    image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]
    #Se introducen a un array cada una y ese array es colocado dentro de la lista
    for i in range(5) :
        X.append(img_to_array(Image.fromarray(image_list[i])))
        y.append(image[i])

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)


X /= 255.0

plt.figure(figsize = (15,5))
for i in range(5) :
    plt.subplot(1,5,i+1)
    plt.imshow(X[i], 'gray')
    plt.title('Label is ' + str(y[i]))
plt.plot()
plt.show()

temp = set(y)
for t in temp :
    print('Occurance count of ' + t + ' : ' + str(len(y[y == t])))
    
temp_df = pandas.DataFrame({'labels' : [t for t in temp], 'Count' : [len(y[y==t]) for t in temp]})

plt.figure(figsize = (15,7))
seaborn.barplot(x = 'labels', y = 'Count', data = temp_df, palette = 'Blues_d')
plt.title('Label distribution in CAPTCHAS', fontsize = 20)
plt.show()

#One hot encoding
y_combine = LabelEncoder().fit_transform(y)
y_one_hot = OneHotEncoder(sparse = False).fit_transform(y_combine.reshape(len(y_combine),1))

print('letter n : ' + str(y[1]))
print('label : ' + str(y_combine[1]))
print('Count : ' + str(len(y_combine[y_combine == y_combine[1]])))

info = {y_combine[i] : y[i] for i in range(len(y))}

print(X.shape)
print(y_one_hot.shape)  # one hot encoded

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size = 0.2, random_state = 1)
y_temp = np.argmax(y_test, axis = 1)
temp = set(y_temp)
temp_df = pandas.DataFrame({'labels' : [info[t] for t in temp], 'Count' : [len(y_temp[y_temp == t]) for t in temp]})
print(temp_df)

plt.figure(figsize = (15,7))
seaborn.barplot(x = 'labels', y = 'Count', data = temp_df, palette = 'Blues_d')
plt.title('Label distribution in test set', fontsize = 20)
plt.show()

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

X_train = np.reshape(X_train, (4160, 40*20*1))

#imbalanced learning
X_train, y_train = SMOTE(sampling_strategy = 'auto', random_state = 1).fit_resample(X_train, y_train)

print(X_train.shape)
print(y_train.shape)

X_train = np.reshape(X_train, (8037, 40, 20, 1))


#imagenes sobresampleadas
plt.figure(figsize = (15,5))

hi = 8030
lo = 5000

for i in range(25) :
    plt.subplot(5,5,i+1)
    x = np.random.randint(lo, hi)
    plt.imshow(X_train[x], 'gray')
    plt.title('Label is ' + str(info[np.argmax(y_train[x])]))
plt.show()

#image data generator
traingen = ImageDataGenerator(rotation_range = 5, width_shift_range = [-2,2])
traingen.fit(X_train)

train_set = traingen.flow(X_train, y_train)

trainX, trainy = train_set.next()

#plot image data
plt.figure(figsize = (15,5))

hi = 32
lo = 0

for i in range(25) :
    plt.subplot(5,5,i+1)
    x = np.random.randint(lo, hi)
    plt.imshow(trainX[x], 'gray')
    plt.title('Label is ' + str(info[np.argmax(trainy[x])]))
plt.show()


model = cnn(128, 32, 16, 32, 32)
model.summary()

checkp = ModelCheckpoint('./result_model.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)

reduce = ReduceLROnPlateau(monitor = 'val_loss', patience = 20, verbose = 1)

print(X_train.shape)
print(y_train.shape)

history = model.fit(traingen.flow(X_train, y_train, batch_size = 32), validation_data = (X_test, y_test), epochs = 150, steps_per_epoch = len(X_train)/32, callbacks = [checkp])


plt.figure(figsize = (20,10))
plt.subplot(2,1,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend(['train loss','val loss'])
plt.title('Loss function wrt epochs')

plt.subplot(2,1,2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train acc' , 'val acc'])
plt.title('Model accuracy wrt Epoch')


model = load_model('./result_model.h5')


pred = model.predict(X_test)


pred = np.argmax(pred, axis = 1)
yres = np.argmax(y_test,axis= 1)


target_name = []
for i in sorted(info) :
    target_name.append(info[i])

print(target_name)
print('Accuracy : ' + str(accuracy_score(yres, pred)))
print(classification_report(yres, pred, target_names = target_name))


# def get_demo (img_path) :
    
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
#     plt.imshow(img, 'gray')
#     plt.axis('off')
#     plt.show()
    
#     img = t_img(img)
#     img = c_img(img)
#     img = d_img(img)
#     img = b_img(img)
    
#     image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]
    
#     plt.imshow(img, 'gray')
#     plt.axis('off')
#     plt.show()
#     Xdemo = []
#     for i in range(5) :
#         Xdemo.append(img_to_array(Image.fromarray(image_list[i])))
    
#     Xdemo = np.array(Xdemo)
#     Xdemo/= 255.0
    
#     ydemo = model.predict(Xdemo)
#     ydemo = np.argmax(ydemo, axis = 1)
    
#     for res in ydemo :
#         print(info[res])
#     print(img_path[-9:])

