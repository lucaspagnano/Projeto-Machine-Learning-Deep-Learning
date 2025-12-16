from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. Preparação dos dados
train_dir = "dataset_train"

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# 2. Definição do modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(5, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 3. Treino
history = model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen
)

# 4. Guardar modelo treinado
model.save("modelo_treinado.h5")
print("Modelo salvo em 'modelo_treinado.h5'")

#5. Graficos treino vs validation
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.title("Evolução da Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("Evolução da Loss")
plt.legend()
plt.tight_layout()
plt.show()