import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Create directory if it doesn't exist
os.makedirs("backend/model", exist_ok=True)

# Now write the file
with open("backend/model/poultry_labels.txt", "w") as f:
    f.write("your data here")


# Paths
DATASET_PATH = r"C:\Users\Chakri\OneDrive\Desktop\disease\chicken_disease\train"
MODEL_PATH = "backend/model/poultry_model.h5"

# Image Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Save labels
with open("backend/model/poultry_labels.txt", "w") as f:
    for label in train_gen.class_indices:
        f.write(label + "\n")

# Transfer Learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save Model
model.save(MODEL_PATH)
print("Model Saved.")
