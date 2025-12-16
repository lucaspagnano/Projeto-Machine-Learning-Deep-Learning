from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam  
import matplotlib.pyplot as plt

# Preparação dos dados 
train_dir = "dataset/train"

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(128, 128), 
    batch_size=32, 
    class_mode="categorical", 
    subset="training", 
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(128, 128), 
    batch_size=32, 
    class_mode="categorical", 
    subset="validation", 
    shuffle=False
)

# --- Definição do modelo CNN (COM VALORES PSO) ---

# Hiperparâmetros encontrados pelo PSO
PSO_LR = 0.000954
PSO_DROPOUT = 0.52
PSO_NEURONS = 162

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(PSO_NEURONS, activation="relu"), 
    layers.Dropout(PSO_DROPOUT),
    layers.Dense(5, activation="softmax")
])

# Criar o otimizador
optimizer = Adam(learning_rate=PSO_LR) 

model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Treino 
history = model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen
)

# Guardar modelo treinado
model.save("modelo_PSO.h5")
print("Modelo otimizado com PSO salvo em 'modelo_PSO.h5'")

# Graficos treino vs validation
model.save("modelo_treinado.h5")
print("Modelo salvo em 'modelo_treinado.h5'")

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