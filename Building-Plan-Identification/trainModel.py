import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Image dimensions and batch size
img_width, img_height = 600, 600
batch_size = 16


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,               # slight rotation up to 5°
    width_shift_range=0.05,         # small horizontal shift
    height_shift_range=0.05,        # small vertical shift
    zoom_range=0.1,                 # zoom in/out by 10%
    brightness_range=[0.9, 1.1],    # subtle brightness changes
    preprocessing_function=lambda x: x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.02)
)




# Validation / Test data just needs rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    'data/trainingData',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    'data/testingData/validationData',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Improved CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),  # reduce overfitting

    layers.Dense(1, activation='sigmoid')
])

# Compile with Adam optimizer and learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks: EarlyStopping and saving best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('earlyStopClassifier_large.h5', save_best_only=True)
]

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=30,  # increase epochs, callbacks will stop if overfitting
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks
)

# Final save
model.save('123classifier_large.h5')