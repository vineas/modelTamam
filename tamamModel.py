#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jt -t chesterish')


# In[1]:


# manggil dataset & extract
from zipfile import ZipFile
file_name= "tamam-dataset.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('Sudah')


# In[1]:


# manggil library
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image


# In[2]:


# memberi label 
data_dir='./tamam-dataset/'

categories = ['Huruf yang tidak dapat dirangkai ke depan', 'Titik yang membentuk lima bunyi huruf', 'Ekor yang dipotong', 'Ekor yg diluruskan', 'Dirangkai tanpa perubahan', 'Ekor yang diluruskan dan segitiga di tengah', 'Merangkai huruf fa qaf', 'Huruf kaf lam', 'Huruf mim ha']

data = []


# In[7]:


# inisialisasi data dan memberi label pada setiap data
def make_data ():
    for category in categories:
        path  = os.path.join(data_dir, category)
        label = categories.index(category)

        for img_name in os.listdir(path):
                image_path=os.path.join(path, img_name) 
                image=cv2.imread(image_path)

                try:
#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (28,28))
                    image = np.array (image, dtype=np.float32)
                    data.append([image, label])
                except Exception as e:
                        pass

    print(len(data))
    
    pik=open('data2.pickle', 'wb') #save data yang sudah diberi label 
    pickle.dump(data, pik)
    pik.close()


    #       cv2.imshow('image sdsdsfsfsd', image)
    #       break
    #     break
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()
make_data ()


# In[3]:


# fungsi load data dari pickle
def load_data():
    pick = open ('data2.pickle', 'rb')
    data = pickle.load(pick)
    pick.close()
    
    np.random.shuffle(data)
    
    feature = []
    labels = []
    
    for img, label in data:
        feature.append(img)
        labels.append(label)
    
    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels)
    
    feature = feature/255.0
    
    return [feature, labels]


# In[4]:


# load data
(feature, labels) = load_data()


# In[5]:


feature


# In[6]:


labels


# In[7]:


# memisahkan dataset
# melakukan pengujian dengan parameter dengan train dataset yang berbeda
x_train, x_test, y_train, y_test= train_test_split(feature, labels, test_size = 0.1)


# In[8]:


input_layer = tf.keras.layers.Input([28,28,3])

conv1=tf.keras.layers.Conv2D(filters = 32, kernel_size=(5,5), padding='Same',
                      activation='relu')(input_layer)

pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool1)

pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv2)

# conv3 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
#                               activation='relu')(pool2)

# pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv3)

# conv4 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
#                               activation='relu')(pool3)

# pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv4)

flt1 = tf.keras.layers.Flatten()(pool2)
dn1 = tf.keras.layers.Dense(512, activation='relu')(flt1)
out = tf.keras.layers.Dense(9, activation='softmax')(dn1)

model = tf.keras.Model(input_layer, out)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


# tain model
model_history = model.fit(x_train, y_train, batch_size = 100, epochs = 50)
validation_data = (x_test,y_test)
# save model
# model.save('mymodel.h5')


# In[10]:


# menghitung accuracy menggunakan confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
ypred = model.predict(x_test)
cnn_accuracy = accuracy_score(y_test, ypred.argmax(axis=-1))
print("CNN Accuracy:",cnn_accuracy)
cnn_cn = confusion_matrix(y_test,ypred.argmax(axis=-1))
plt.subplots(figsize=(20,16))
sns.heatmap(cnn_cn,annot=True,fmt="1d",cmap="Blues",cbar=True,xticklabels=categories,yticklabels=categories)
plt.xlabel("Predicted",fontsize=15)
plt.ylabel("Actual",fontsize=15)
# plt.savefig('model60_e40.png')
plt.show()


# In[11]:


model.summary()


# In[12]:


plt.plot(model_history.history['accuracy'], label='accuracy')
# plt.plot(model_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(x_test, verbose=2)


# In[13]:


score = model.evaluate(x_test, y_test)
print('Test accuarcy: %0.2f%%' % (score[0] * 100))


# In[14]:


# save model
model.save('model90_e10.h5')


# In[15]:


cnn_accuracy


# In[16]:


# Deteksi
ypred = model.predict(x_test)


# In[17]:


y_test


# In[18]:


ypred[0]


# In[19]:


from sklearn.metrics import classification_report
print(classification_report(y_test, ypred.argmax(axis=-1),target_names= categories))


# In[20]:


model_load = tf.keras.models.load_model('model90_e10.h5')
print('loaded')


# In[21]:


path = "hijaiyah/ba-kecil.jpg"
img = image.load_img(path, target_size=(28,28))
imgplot = plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model_load.predict(images, batch_size=10)


# In[22]:


print(classes)


# In[23]:


m = 0.00000
for category in range(len(categories)):
  if m < classes[0][category]:
    m = category
if m==0:
  print("Kelas : Huruf yang tidak dapat dirangkai ke depan")
elif m==1:
  print("Kelas : Titik yang membentuk lima bunyi huruf")
elif m==2:
  print("Kelas : Ekor yang dipotong")
elif m==3:
  print("Kelas : Ekor yg diluruskan")
elif m==4:
  print("Kelas : Dirangkai tanpa perubahan")
elif m==5:
  print("Kelas : Ekor yang diluruskan dan segitiga di tengah")
elif m==6:
  print("Kelas : Merangkai huruf fa qaf")     
elif m==7:
  print("Kelas : Huruf kaf lam")
elif m==8:
  print("Kelas : Huruf mim ha")
else:
  print("Kelas : tidak terdeteksi")


# In[33]:


x_train, x_test, y_train, y_test= train_test_split(feature, labels, test_size = 0.2)


# In[34]:


input_layer = tf.keras.layers.Input([28,28,3])

conv1=tf.keras.layers.Conv2D(filters = 32, kernel_size=(5,5), padding='Same',
                      activation='relu')(input_layer)

pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool1)

pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool2)

pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool3)

pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv4)

flt1 = tf.keras.layers.Flatten()(pool4)
dn1 = tf.keras.layers.Dense(512, activation='relu')(flt1)
out = tf.keras.layers.Dense(9, activation='softmax')(dn1)

model = tf.keras.Model(input_layer, out)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[35]:


model_history = model.fit(x_train, y_train, batch_size = 100, epochs = 10)
validation_data = (x_test,y_test)


# In[36]:


# menghitung accuracy menggunakan confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
ypred = model.predict(x_test)
cnn_accuracy = accuracy_score(y_test, ypred.argmax(axis=-1))
print("CNN Accuracy:",cnn_accuracy)
cnn_cn = confusion_matrix(y_test,ypred.argmax(axis=-1))
plt.subplots(figsize=(20,16))
sns.heatmap(cnn_cn,annot=True,fmt="1d",cmap="Blues",cbar=True,xticklabels=categories,yticklabels=categories)
plt.xlabel("Predicted",fontsize=15)
plt.ylabel("Actual",fontsize=15)
# plt.savefig('model60_e40.png')
plt.show()


# In[37]:


plt.plot(model_history.history['accuracy'], label='accuracy')
# plt.plot(model_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[38]:


cnn_accuracy


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_test, ypred.argmax(axis=-1),target_names= categories))


# In[40]:


x_train, x_test, y_train, y_test= train_test_split(feature, labels, test_size = 0.3)


# In[41]:


input_layer = tf.keras.layers.Input([28,28,3])

conv1=tf.keras.layers.Conv2D(filters = 32, kernel_size=(5,5), padding='Same',
                      activation='relu')(input_layer)

pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool1)

pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool2)

pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool3)

pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv4)

flt1 = tf.keras.layers.Flatten()(pool4)
dn1 = tf.keras.layers.Dense(512, activation='relu')(flt1)
out = tf.keras.layers.Dense(9, activation='softmax')(dn1)

model = tf.keras.Model(input_layer, out)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[42]:


model_history = model.fit(x_train, y_train, batch_size = 100, epochs = 10)
validation_data = (x_test,y_test)


# In[43]:


# menghitung accuracy menggunakan confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
ypred = model.predict(x_test)
cnn_accuracy = accuracy_score(y_test, ypred.argmax(axis=-1))
print("CNN Accuracy:",cnn_accuracy)
cnn_cn = confusion_matrix(y_test,ypred.argmax(axis=-1))
plt.subplots(figsize=(20,16))
sns.heatmap(cnn_cn,annot=True,fmt="1d",cmap="Blues",cbar=True,xticklabels=categories,yticklabels=categories)
plt.xlabel("Predicted",fontsize=15)
plt.ylabel("Actual",fontsize=15)
# plt.savefig('model60_e40.png')
plt.show()


# In[44]:


plt.plot(model_history.history['accuracy'], label='accuracy')
# plt.plot(model_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[45]:


cnn_accuracy


# In[46]:


from sklearn.metrics import classification_report
print(classification_report(y_test, ypred.argmax(axis=-1),target_names= categories))


# In[47]:


x_train, x_test, y_train, y_test= train_test_split(feature, labels, test_size = 0.4)

input_layer = tf.keras.layers.Input([28,28,3])

conv1=tf.keras.layers.Conv2D(filters = 32, kernel_size=(5,5), padding='Same',
                      activation='relu')(input_layer)

pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = tf.keras.layers.Conv2D(filters= 64, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool1)

pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool2)

pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv3)

conv4 = tf.keras.layers.Conv2D(filters= 96, kernel_size=(3,3), padding='Same',
                              activation='relu')(pool3)

pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(conv4)

flt1 = tf.keras.layers.Flatten()(pool4)
dn1 = tf.keras.layers.Dense(512, activation='relu')(flt1)
out = tf.keras.layers.Dense(9, activation='softmax')(dn1)

model = tf.keras.Model(input_layer, out)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_history = model.fit(x_train, y_train, batch_size = 100, epochs = 10)
validation_data = (x_test,y_test)


# In[48]:


# menghitung accuracy menggunakan confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
ypred = model.predict(x_test)
cnn_accuracy = accuracy_score(y_test, ypred.argmax(axis=-1))
print("CNN Accuracy:",cnn_accuracy)
cnn_cn = confusion_matrix(y_test,ypred.argmax(axis=-1))
plt.subplots(figsize=(20,16))
sns.heatmap(cnn_cn,annot=True,fmt="1d",cmap="Blues",cbar=True,xticklabels=categories,yticklabels=categories)
plt.xlabel("Predicted",fontsize=15)
plt.ylabel("Actual",fontsize=15)
# plt.savefig('model60_e40.png')
plt.show()


# In[49]:


plt.plot(model_history.history['accuracy'], label='accuracy')
# plt.plot(model_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[50]:


cnn_accuracy


# In[51]:


from sklearn.metrics import classification_report
print(classification_report(y_test, ypred.argmax(axis=-1),target_names= categories))


# In[ ]:





# In[ ]:





# In[ ]:


from ipywidgets import FileUpload
upload = FileUpload()
upload


# In[ ]:


import matplotlib.image as mpimg
from ipywidgets import FileUpload
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.preprocessing.image import ImageDataGenerator


upload = FileUpload()
upload

for fn in upload.keys():
  # predicting images
    path = fn
    img = image.load_img(path, target_size=(28,28))
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    if classes[0,0]!=1:
        print("Huruf yang tidak dapat dirangkai ke depan")
    elif classes[0,1]!=1:
        print("Titik yang membentuk lima bunyi huruf")
    elif classes[0,2]!=1:
        print("Ekor yang dipotong")
    elif classes[0,3]!=1:
        print("Ekor yg diluruskan")
    elif classes[0,4]!=1:
        print("Dirangkai tanpa perubahan")
    elif classes[0,5]!=1:
        print("Ekor yang diluruskan dan segitiga di tengah")
    elif classes[0,6]!=1:
        print("Merangkai huruf fa qaf")     
    elif classes[0,7]!=1:
        print("Huruf kaf lam")
    elif classes[0,8]!=1:
        print("Huruf mim ha")

    else:
        print("tidak terdeteksi")


# In[ ]:


model.predict(img)


# In[ ]:


# (feature, labels)=load_data()


# In[ ]:


x_train, x_test, y_train, y_test= train_test_split(feature, labels, test_size=0.1)

categories=['1', '2', '3', '4', '5', '6', '7', '8', '9']

model= tf.keras.models.load_model('mymodel.h5')

# model.evaluate(x_test, y_test, verbose = 1)

prediction = model.predict(x_test)

plt.figure(figsize=(9,9))
print('1. Huruf yang tidak dapat dirangkai ke depan')
print('2. Titik yang membentuk lima bunyi huruf')
print('3. Ekor yang dipotong') 
print('4. Ekor yg diluruskan') 
print('5. Dirangkai tanpa perubahan') 
print('6. Ekor yang diluruskan dan segitiga di tengah') 
print('7. Merangkai huruf fa qaf') 
print('8. Huruf kaf lam') 
print('9. Huruf mim ha')

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i])
    plt.xlabel('actual: '+categories[y_test[i]]+'\n'+'predict: '+
              categories[np.argmax(prediction[i])])
    
    plt.xticks([])
plt.show()


# In[ ]:




