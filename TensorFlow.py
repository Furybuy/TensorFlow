import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#to run the next line on 
#windows use: set TF_ENABLE_ONEDNN_OPTS=0 
#Linux use: export TF_ENABLE_ONEDNN_OPTS=0

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#Dividing by 255.0 normalizes the pixel values.
normalizer = 255.0
train_images = train_images / normalizer
test_images = test_images / normalizer

number_pixels = 28
number_neurons_mid_layer = 50
number_neurons_end_layer = 10
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(number_pixels, number_pixels)),
    keras.layers.Dense(number_neurons_mid_layer, activation=tf.nn.relu),
    keras.layers.Dense(number_neurons_end_layer, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

history = model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'Test acc: {test_acc}')

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia (Treino)', marker='o')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(
    history.history['loss'],
    label='Perda (Treino)',
    color='orange',
    marker='o',
)
plt.title('Evolução da Perda (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.show()