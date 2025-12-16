import numpy as np
from SwarmPackagePy import pso, gwo
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import time 

# --- Configurações dos Geradores de Dados ---
train_dir = "dataset/train"

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

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

# --- "Função de Fitness" da CNN ---
EPOCHS_POR_TESTE = 5 

def cnn_fitness_function(solution):
    
    learning_rate = solution[0]
    dropout_rate  = solution[1]
    dense_neurons = int(solution[2]) 

    print(f"\n--- A Testar: LR={learning_rate:.6f} | Dropout={dropout_rate:.2f} | Neurons={dense_neurons} ---")

    try:
        tf.keras.backend.clear_session() 
        
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, (3,3), activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(dense_neurons, activation="relu"), 
            layers.Dropout(dropout_rate),                   
            layers.Dense(5, activation="softmax")
        ])

        optimizer = Adam(learning_rate=learning_rate) 
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(
            train_gen,
            epochs=EPOCHS_POR_TESTE,
            validation_data=val_gen,
            verbose=0  
        )

        val_accuracy = history.history['val_accuracy'][-1]
        fitness_score = 1.0 - val_accuracy
        
        print(f"--- Resultado: Val_Acc: {val_accuracy:.4f} | Custo (Fitness): {fitness_score:.4f} ---")

    except Exception as e:
        print(f"!!! Erro ao treinar: {e}. A descartar solução. !!!")
        fitness_score = 10.0 
    
    return fitness_score

# --- Configurações da Otimização Swarm ---
D = 3 
lb = [0.0001, 0.2, 64]   
ub = [0.01,   0.6, 256]  

N_AGENTS = 8     
N_ITERATIONS = 8 
# (Total de redes a treinar = 8 * 8 * 2 = 128)

print(f"--- Iniciando Otimização da CNN ---")
print(f"Dimensões (Hiperparâmetros): {D}")
print(f"Agentes: {N_AGENTS} | Iterações: {N_ITERATIONS}")
print(f"Épocas por avaliação: {EPOCHS_POR_TESTE}")
print(f"Total de redes a treinar: {N_AGENTS * N_ITERATIONS * 2} (GWO + PSO)")
print("--------------------------------------------------\n")

# --- Executar GWO ---
start_time_gwo = time.time()
print("A executar Otimização com GWO...")

gwo_optimizer = gwo(N_AGENTS, cnn_fitness_function, lb, ub, D, N_ITERATIONS)

best_solution_gwo = gwo_optimizer._sw__Gbest
best_fitness_gwo = cnn_fitness_function(best_solution_gwo) 

end_time_gwo = time.time()
print(f"\n--- OTIMIZAÇÃO GWO CONCLUÍDA (Tempo: { (end_time_gwo - start_time_gwo) / 60 :.2f} minutos) ---")
print(f"Melhor Custo (1 - Val_Acc): {best_fitness_gwo:.4f}")
print("Melhores Hiperparâmetros (GWO):")
print(f"  Learning Rate: {best_solution_gwo[0]:.6f}")
print(f"  Dropout Rate:  {best_solution_gwo[1]:.2f}")
print(f"  Dense Neurons: {int(best_solution_gwo[2])}")
print("--------------------------------------------------\n")


# --- Executar PSO ---
start_time_pso = time.time()
print("A executar Otimização com PSO...")

pso_optimizer = pso(N_AGENTS, cnn_fitness_function, lb, ub, D, N_ITERATIONS)

best_solution_pso = pso_optimizer._sw__Gbest
best_fitness_pso = cnn_fitness_function(best_solution_pso) 

end_time_pso = time.time()
print(f"\n--- OTIMIZAÇÃO PSO CONCLUÍDA (Tempo: { (end_time_pso - start_time_pso) / 60 :.2f} minutos) ---")
print(f"Melhor Custo (1 - Val_Acc): {best_fitness_pso:.4f}")
print("Melhores Hiperparâmetros (PSO):")
print(f"  Learning Rate: {best_solution_pso[0]:.6f}")
print(f"  Dropout Rate:  {best_solution_pso[1]:.2f}")
print(f"  Dense Neurons: {int(best_solution_pso[2])}")
print("--------------------------------------------------\n")