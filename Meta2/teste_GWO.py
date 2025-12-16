import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# 1. Carregar modelo treinado
model = tf.keras.models.load_model("modelo_GWO.h5")
print("Modelo carregado com sucesso!")

# 2. Preparar dados de teste
test_dir = "dataset/test"

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

# 3. Avaliação no conjunto de teste
test_loss, test_acc = model.evaluate(test_gen)
print(f"Accuracy no conjunto de teste: {test_acc:.4f}")

# 4. Relatório e Matriz de Confusão
y_prob = model.predict(test_gen)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_gen.classes

class_indices = test_gen.class_indices
classes = [cls for cls, idx in sorted(class_indices.items(), key=lambda x: x[1])]

print(classification_report(y_true, y_pred, target_names=classes))

cm = confusion_matrix(y_true, y_pred)

specificities = []
for i in range(len(cm)):
    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # tudo exceto linha e coluna i
    fp = np.sum(np.delete(cm, i, axis=0)[:, i])  # coluna i exceto linha i
    specificity = tn / (tn + fp)
    specificities.append(specificity)

specificity_macro = np.mean(specificities)
print(f"Especificidade média (macro): {specificity_macro:.4f}")

# One-hot encode das classes reais
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
# AUC média (macro)
auc_macro = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
print(f"AUC média (macro): {auc_macro:.4f}")

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão (contagens)")
plt.tight_layout()
plt.show()