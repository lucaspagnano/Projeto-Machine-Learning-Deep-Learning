import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# 1. CONFIGURAÇÕES
# =============================================================================
TEST_DIR = 'dataset/test'
MODEL_FILE = 'modelo_mobilenet_25.h5' 
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =============================================================================
# 2. PREPARAR DADOS DE TESTE
# =============================================================================
print("--- A carregar dados de teste ---")

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# =============================================================================
# 3. CARREGAR MODELO E PREVER
# =============================================================================
print(f"--- A carregar modelo: {MODEL_FILE} ---")
model = load_model(MODEL_FILE)

print("--- A realizar previsões... ---")
# Predict devolve probabilidades
Y_pred = model.predict(test_generator, verbose=1)
# Converter probabilidades em classes (0, 1, 2, ...)
y_pred = np.argmax(Y_pred, axis=1)

# Obter as classes reais
y_true = test_generator.classes

# Obter nomes das classes
class_labels = list(test_generator.class_indices.keys())

# =============================================================================
# 4. GERAR METRÍCAS E MATRIZ
# =============================================================================
print("\n--- Relatório de Classificação ---")
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Guardar relatório em texto para o documento
with open('relatorio_report_25.txt', 'w') as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusão - MobileNetV2 (Transfer Learning)')
plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.savefig('matriz_confusao_25.png') # Guarda a imagem para o relatório
plt.show()

