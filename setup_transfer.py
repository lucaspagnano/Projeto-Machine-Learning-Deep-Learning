import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
import os

# --- 1. Configurações e Verificação de Dados ---
train_dir = "dataset/train"
# Define o tamanho que a MobileNet espera
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32

# Verificar balanceamento (Requisito do enunciado)
print("--- Verificação de Balanceamento ---")
classes = os.listdir(train_dir)
for c in classes:
    path = os.path.join(train_dir, c)
    if os.path.isdir(path):
        num = len(os.listdir(path))
        print(f"Classe '{c}': {num} imagens")
print("------------------------------------\n")

# --- 2. Preparar os Geradores de Dados (Com Pré-processamento da MobileNet) ---
# IMPORTANTE: Em vez de rescale=1./255, usamos a função da própria MobileNet
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # <-- Mudança Crítica para Transfer Learning
    validation_split=0.2,
    # Data Augmentation simples para ajudar no balanceamento se necessário
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# --- 3. Construir o Modelo de Transfer Learning ---
def build_transfer_model(learning_rate, dropout_rate, dense_neurons, num_classes):
    # 1. Carregar a "Base" (O cérebro pré-treinado)
    base_model = MobileNetV2(
        weights='imagenet',  # Usar conhecimento prévio
        include_top=False,   # Não incluir a camada final de classificação (1000 classes)
        input_shape=(224, 224, 3)
    )
    
    # 2. Congelar a base (Não queremos estragar o que a rede já sabe)
    base_model.trainable = False 
    
    # 3. Criar a nova "Cabeça" (Top Layers) 
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D()) # Resume a informação da base
    model.add(layers.Dense(dense_neurons, activation='relu')) # Camada densa que vamos otimizar
    model.add(layers.Dropout(dropout_rate)) # Dropout que vamos otimizar
    model.add(layers.Dense(num_classes, activation='softmax')) # 5 Classes de lixo
    
    # 4. Compilar
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --- 4. Teste Rápido ---
print("\n--- A testar a construção do modelo... ---")
try:
    model = build_transfer_model()
    print("Modelo MobileNetV2 carregado com sucesso!")
    model.summary()
    print("\nTeste de 1 época para garantir que tudo corre bem...")
    model.fit(train_gen, epochs=1, validation_data=val_gen)
    print("Sucesso! O pipeline está pronto.")
except Exception as e:

    print(f"Erro: {e}")
