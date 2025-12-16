import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from SwarmPackagePy import gwo

# =============================================================================
# 0. CONFIGURAÇÃO DOS DADOS (Requisito: preprocess_input)
# =============================================================================
TRAIN_DIR = "dataset/train"

# Configurar geradores com a normalização da MobileNetV2
# Usamos apenas um gerador com validation_split para garantir consistência
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # CRUCIAL para MobileNetV2
    validation_split=0.2, 
    rotation_range=20,
    horizontal_flip=True
    # Adicione outros augmentations se necessário
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224), # Tamanho nativo da MobileNetV2
    batch_size=32,
    class_mode='categorical',
    subset='training', 
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation', 
    shuffle=False
)

# =============================================================================
# 1. VARIÁVEIS GLOBAIS E LIMITES
# =============================================================================
# Lista global para guardar o histórico para o Excel
results_log = []

# Identificador do algoritmo a correr no momento (para o Excel)
CURRENT_ALGO = "None" 

# Limites [Neurónios, Dropout, Learning Rate]
LB = [32,    0.0,  0.00001]
UB = [512,   0.5,  0.01]

# Configurações de execução
N_AGENTS = 5     # Tamanho da população (lobos)
N_ITER = 5       # Número de iterações
EPOCHS = 5       

# =============================================================================
# 2. FUNÇÃO DE FITNESS (1 - Accuracy)
# =============================================================================
def fitness_function(params):
    """
    Função objetivo a ser MINIMIZADA pelo GWO.
    Retorna (1 - val_accuracy).
    """
    # 1. Descodificar Parâmetros
    n_neurons = int(params[0])   # Cast para int
    dropout_rate = params[1]
    learning_rate = params[2]
    
    # Garantir limites de segurança caso o algoritmo passe um pouco
    n_neurons = max(int(LB[0]), min(int(UB[0]), n_neurons))
    dropout_rate = max(LB[1], min(UB[1], dropout_rate))
    learning_rate = max(LB[2], min(UB[2], learning_rate))

    # Limpeza de sessão para evitar erro de memória
    tf.keras.backend.clear_session() 

    # 2. Construir Modelo MobileNetV2
    base_model = MobileNetV2(weights='imagenet', 
                             include_top=False, 
                             input_shape=(224, 224, 3))
    base_model.trainable = False # Congelar pesos base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Camadas a Otimizar
    x = Dense(n_neurons, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Camada Final
    # No teu caso são 5 classes, então categorical_crossentropy e softmax
    num_classes = 5 
    activation = 'softmax'
    loss_type = 'categorical_crossentropy'
    
    outputs = Dense(num_classes, activation=activation)(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    # 3. Compilar e Treinar
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss_type,
                  metrics=['accuracy'])
    
    # Treino curto
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        verbose=0) # Silencioso na consola
    
    # 4. Calcular Fitness
    best_val_acc = max(history.history['val_accuracy'])
    fitness_val = 1.0 - best_val_acc # Minimização
    
    # 5. Guardar no Log (Excel)
    # Apanhamos os dados "em tempo real"
    results_log.append({
        'Algorithm': CURRENT_ALGO,
        'Neurons': n_neurons,
        'Dropout': dropout_rate,
        'Learning_Rate': learning_rate,
        'Accuracy': best_val_acc,
        'Fitness (1-Acc)': fitness_val
    })
    
    # Print de progresso
    print(f"[{CURRENT_ALGO}] Neurons={n_neurons}, Drop={dropout_rate:.2f}, LR={learning_rate:.5f} -> Acc={best_val_acc:.4f}")

    return fitness_val

# =============================================================================
# 3. EXECUÇÃO GWO (SwarmPackagePy)
# =============================================================================
print("\n--- A INICIAR GWO (SwarmPackagePy) ---")
CURRENT_ALGO = "GWO"

# A biblioteca usa: gwo(n, function, lb, ub, dimension, iteration)
gwo_algo = gwo(N_AGENTS, fitness_function, LB, UB, 3, N_ITER)

best_gwo_params = gwo_algo._sw__Gbest
print(f"Melhor GWO: {best_gwo_params}")

# =============================================================================
# 4. EXECUÇÃO RANDOM SEARCH (Para Comparação)
# =============================================================================
print("\n--- A INICIAR RANDOM SEARCH ---")
CURRENT_ALGO = "Random Search"

# Para ser justo, fazemos o mesmo número de avaliações que o GWO
total_evals = N_AGENTS * N_ITER 

best_rs_fitness = 1.0
best_rs_params = []

for i in range(total_evals):
    # Gerar parâmetros aleatórios
    r_neurons = np.random.randint(LB[0], UB[0] + 1)
    r_dropout = np.random.uniform(LB[1], UB[1])
    r_lr = np.random.uniform(LB[2], UB[2])
    
    params = [r_neurons, r_dropout, r_lr]
    
    # Chamar a mesma função de fitness
    fit = fitness_function(params)
    
    if fit < best_rs_fitness:
        best_rs_fitness = fit
        best_rs_params = params

print(f"Melhor Random Search: {best_rs_params} | Fitness: {best_rs_fitness}")
