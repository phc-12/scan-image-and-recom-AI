import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ['casual', 'formal', 'streetwear', 'bohemian', 'vintage', 'preppy', 'sporty', 'business', 'romantic', 'sexy']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
    
plt.show()


training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

models = models.Sequential()
models.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
models.add(layers.MaxPooling2D((2,2)))
models.add(layers.Conv2D(64, (3,3), activation = 'relu'))
models.add(layers.MaxPooling2D((2,2)))
models.add(layers.Conv2D(64, (3,3), activation = 'relu'))
models.add(layers.Flatten())
models.add(layers.Dense(64, activation = 'relu'))
models.add(layers.Dense(10, activation = 'softmax'))

models.compile(optimize= 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
models.fit(training_images, training_labels, epochs = 10, validation_data = (testing_images, testing_labels))

loss, accuracy = models.evaluate(testing_images, testing_labels)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

models.save('fashion_classifier_model.h5')
model = models.load_model('fashion_classifier_model.h5')

img = cv.imread('formal-wear-1.webp.jpg')

plt.imshow(img, cmap=plt.cm.binary)

plt.show()