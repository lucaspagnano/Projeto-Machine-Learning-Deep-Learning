import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# Importa a função ajustada do seu ficheiro
from setup_transfer import build_transfer_model 

# =============================================================================
# 1. CONFIGURAÇÕES E MELHORES HIPERPARÂMETROS
# =============================================================================
TRAIN_DIR = "dataset/train"  # Caminho para as pastas de treino
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50 

# --- PREENCHA COM OS SEUS DADOS DO EXCEL ---
BEST_NEURONS = 497    
BEST_DROPOUT = 0.17   
BEST_LR      = 0.00047  

# =============================================================================
# 2. PREPARAÇÃO DOS DADOS (DATAFRAME)
# =============================================================================
def create_dataframe(directory):
    filepaths = []
    labels = []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"A pasta {directory} não existe!")
        
    classes = os.listdir(directory)
    for class_name in classes:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                filepaths.append(os.path.join(class_path, filename))
                labels.append(class_name)
    return pd.DataFrame({'filename': filepaths, 'class': labels})

full_df = create_dataframe(TRAIN_DIR)
num_classes = len(full_df['class'].unique())
print(f"Total de imagens encontradas: {len(full_df)}")
print(f"Número de Classes: {num_classes}")

# Separar Validação FIXA (20%) para ser justo em todos os testes
train_df_full, val_df = train_test_split(
    full_df, 
    test_size=0.2, 
    stratify=full_df['class'], # Mantém o balanceamento das classes
    random_state=42
)

# Gerador de Validação (O mesmo para todos os modelos)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='filename', y_col='class',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# =============================================================================
# 3. CICLO DE TREINO (100%, 50%, 25%)
# =============================================================================
fractions = [1.0, 0.5, 0.25]
results = []

print("\n--- INICIANDO A GERAÇÃO DOS MODELOS ---")

for frac in fractions:
    percent_str = int(frac * 100)
    print(f"\n>>> TREINANDO COM {percent_str}% DOS DADOS DE TREINO <<<")
    
    # 1. Amostrar dados (Mantendo estratificação/balanceamento)
    if frac < 1.0:
        train_subset = train_df_full.groupby('class', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=42)
        )
    else:
        train_subset = train_df_full

    print(f"Imagens usadas: {len(train_subset)}")
    
    # 2. Criar Gerador de Treino
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    train_gen = train_datagen.flow_from_dataframe(
        train_subset, x_col='filename', y_col='class',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # 3. Construir Modelo (Passando num_classes explicitamente)
    model = build_transfer_model(BEST_LR, BEST_DROPOUT, BEST_NEURONS, num_classes)
    
    # 4. Configurar Checkpoint para SALVAR O .H5
    model_filename = f"modelo_mobilenet_{percent_str}.h5"
    
    callbacks = [
        # Salva o modelo sempre que a val_accuracy melhorar
        ModelCheckpoint(model_filename, monitor='val_accuracy', save_best_only=True, verbose=1),
        # Pára se não melhorar em 3 épocas (opcional, poupa tempo)
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    ]
    
    # 5. Treinar
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # 6. Registar Resultados
    best_acc = max(history.history['val_accuracy'])
    results.append({
        'Dataset_Size': f"{percent_str}%",
        'Num_Images': len(train_subset),
        'Best_Val_Accuracy': best_acc,
        'Model_File': model_filename
    })
    print(f"Modelo salvo como: {model_filename} (Acc: {best_acc:.4f})")

# =============================================================================
# 4. GUARDAR RELATÓRIO FINAL
# =============================================================================
df_res = pd.DataFrame(results)
df_res.to_excel("Resultados_Analise_Reducao.xlsx", index=False)

print("\n--- PROCESSO CONCLUÍDO ---")
print("Ficheiros gerados:")
print(df_res[['Dataset_Size', 'Model_File', 'Best_Val_Accuracy']])