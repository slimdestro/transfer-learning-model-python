import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

#im using vgg16 as source here
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#freezing the layers
for layer in vgg_model.layers:
    layer.trainable = False

#adding custom layers
x = vgg_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

#creating the model
model = Model(inputs=vgg_model.input, outputs=predictions)

#compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

#testing the model
model.evaluate(x_test, y_test)