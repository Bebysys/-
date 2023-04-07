import tensorflow as tf
from tensorflow import keras
import tensorflow.keras
# библиотека для вывода изображений
import matplotlib.pyplot as plt
%matplotlib inline
# -- Импорт для построения модели: --
# импорт слоев
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
# импорт модели
from tensorflow.keras.models import Sequential
# импорт оптимайзера
from tensorflow.keras.optimizers import Adam
# Импортируем набор данных MNIST
from tensorflow.keras.datasets import mnist
# загружаем тренировочные и тестовые данные
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
print(x_train[0].shape,x_train[0].dtype)
print(x_train[0])
print(y_train[0])
# Выведем на экран хранящееся в X_train[0] зображение
plt.imshow(x_train[0], cmap='binary')
plt.axis('off')
# Преобразование данных в матрицах изображений
# X_train.max() возвращает значение 255
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()
# Преобразуем целевые значения методом «one-hot
#encoding»
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# Создаем модель

model = Sequential([

layers.Dense(32, activation='relu',

input_shape=(x_train[0].shape)),

layers.Dense(64, activation='relu'),

layers.Dense(128, activation='relu'),

layers.Dense(256, activation='relu'),

layers.Dense(512, activation='relu'),

layers.Flatten(),

layers.Dense(10, activation='sigmoid')

])

# Выведем полученную модель на экран

model.summary()
#Компиляция модели

model.compile(loss='binary_crossentropy',

optimizer = Adam(lr=0.00024),

metrics = ['binary_accuracy'])
# Функция ранней остановки
stop =

tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=4)
# Запускаем обучение модели

history = model.fit(x_train, y_train, batch_size=500,

verbose=1,

epochs= 5, validation_split =

0.2, callbacks=[stop])
# Предсказываем результат для тестовой выборки

pred = model.predict(x_test)
print(pred[0])

for i in range(len(pred)):

    for j in range(10):

      if(pred[i][j]>0.5):

         pred[i][j]=1

      else:

         pred[i][j]=0


print(pred[3], y_test[3])

# 07.04.2023p.
