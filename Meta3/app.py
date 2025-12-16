import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# =============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Reciclagem Inteligente AI",
    page_icon="♻️",
    layout="centered"
)

# Título e Cabeçalho
st.title("♻️ Classificador de Resíduos")
st.markdown('Esta aplicação utiliza Inteligência Artificial (**MobileNetV2** com Transfer Learning) para identificar o tipo de material reciclável numa imagem.')

# =============================================================================
# 2. DEFINIÇÃO DAS CLASSES
# =============================================================================
# A ordem tem de ser alfabética (igual às pastas de treino)
CLASS_NAMES = ['Metal', 'Organico', 'Papel', 'Plastico', 'Vidro']

# =============================================================================
# 3. CARREGAR O MODELO
# =============================================================================
@st.cache_resource
def load_learner():
    model_path = "modelo_mobilenet_100.h5"
    if not os.path.exists(model_path):
        st.error(f"Erro: Não encontrei o ficheiro '{model_path}' na pasta.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

model = load_learner()

# =============================================================================
# 4. FUNÇÃO DE PREVISÃO
# =============================================================================
def predict_image(image_file):
    # 1. Abrir e redimensionar a imagem para 224x224
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    
    # 2. Converter para array
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Batch de 1
    
    # 3. Pré-processamento (MobileNetV2 espera intervalo [-1, 1])
    processed_img = preprocess_input(img_array)
    
    # 4. Previsão
    predictions = model.predict(processed_img)
    
    # Resultados
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    return CLASS_NAMES[class_idx], confidence, predictions[0]

# =============================================================================
# 5. INTERFACE PRINCIPAL (UPLOAD + CÂMARA)
# =============================================================================
st.sidebar.header("Sobre o Projeto")
st.sidebar.info(f"""
**Modelo:** MobileNetV2
**Classes:** {len(CLASS_NAMES)}
**Performance:** ~95% Accuracy
""")

# --- CRIAÇÃO DAS ABAS ---
tab1, tab2 = st.tabs(["📁 Carregar Imagem", "📸 Tirar Foto"])

image_buffer = None

# Aba 1: Upload de Ficheiro
with tab1:
    uploaded_file = st.file_uploader("Escolhe uma imagem...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_buffer = uploaded_file

# Aba 2: Input da Câmara
with tab2:
    camera_file = st.camera_input("Tira uma foto ao resíduo")
    if camera_file:
        image_buffer = camera_file

# --- PROCESSAMENTO (Comum às duas abas) ---
if image_buffer is not None and model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_buffer, caption='Imagem a Analisar', use_container_width=True)
    
    with col2:
        st.write("⏳ A analisar...")
        
        # Fazer a previsão
        label, confidence, all_probs = predict_image(image_buffer)
        
        # Mostrar Classificação
        st.success(f"**Resultado:** {label}")
        
        # Mostrar Confiança com cores dinâmicas
        if confidence > 0.85:
            st.metric(label="Confiança", value=f"{confidence*100:.1f}%", delta="Alta Certeza")
        elif confidence > 0.60:
            st.metric(label="Confiança", value=f"{confidence*100:.1f}%", delta="Moderada", delta_color="off")
        else:
            st.metric(label="Confiança", value=f"{confidence*100:.1f}%", delta="Incerteza", delta_color="inverse")
            st.warning("Atenção: A confiança é baixa. Verifica a iluminação.")
        
        st.progress(int(confidence * 100))

    # Gráfico de Probabilidades
    st.write("---")
    st.subheader("Detalhes da Probabilidade")
    st.bar_chart({
        "Classe": CLASS_NAMES,
        "Probabilidade": all_probs
    })

elif model is None:
    st.warning("O modelo não está carregado.")