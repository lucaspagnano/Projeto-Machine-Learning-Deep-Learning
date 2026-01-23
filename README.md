# ♻️ Classificador Inteligente de Resíduos (Recycling AI)

> **Unidade Curricular:** Inteligência Computacional<br>
> **Instituição:** ISEC - Instituto Superior de Engenharia de Coimbra  <br>
> **Ano Letivo:** 2025/2026

## 📌 Visão Geral do Projeto
Este projeto consiste no desenvolvimento de um sistema de Visão Computacional baseado em **Deep Learning** para a classificação automática de resíduos recicláveis. O sistema final utiliza **Transfer Learning** (MobileNetV2), otimização de hiperparâmetros via **Swarm Intelligence** (GWO) e uma interface Web para utilização em tempo real.<br>

O projeto foi desenvolvido em três fases distintas, evoluindo de uma abordagem básica para uma solução robusta e otimizada:

**Fase I (Meta I):** Análise do problema, recolha de dataset e desenvolvimento de modelos iniciais (CNN/MLP) "treinados do zero". <br>
**Fase II (Meta II):** Investigação e implementação de algoritmos de inteligência de enxame (**Swarm Intelligence**) para a otimização automática de hiperparâmetros da rede.<br>
**Fase III (Meta III):** Implementação final utilizando **Transfer Learning** (MobileNetV2), análise de robustez com redução de dados e *deployment* numa aplicação Web.<br>

---

## 🚀 Funcionalidades Principais

**Arquitetura MobileNetV2:** Utilização de uma rede pré-treinada na ImageNet para extração de características (Feature Extraction), garantindo leveza e eficiência.<br>
**Otimização com GWO (Grey Wolf Optimizer):** Ajuste automático de neurónios, *dropout* e *learning rate* utilizando inteligência coletiva, superando a pesquisa aleatória (Random Search).<br>
**Robustez a Dados Reduzidos:** Validação da eficácia do modelo mesmo com apenas **25%** do dataset original, demonstrando o poder do Transfer Learning.<br>
**Aplicação Web (Streamlit):** Interface gráfica para classificação em tempo real via **Upload de Imagem** ou **Câmara**.<br>

---

## 📊 Dataset e Classes

O modelo foi treinado para distinguir **5 classes** de resíduos:
1.  **Metal**
2.  **Orgânico**
3.  **Papel**
4.  **Plástico**
5.  **Vidro**

**Resultados Finais (Conjunto de Teste Independente):**
* **Accuracy Global:** ~95%
* **F1-Score:** Consistente acima de 0.94 para todas as classes.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Deep Learning:** TensorFlow / Keras
* **Otimização:** SwarmPackagePy (GWO)
* **Interface:** Streamlit
* **Processamento de Dados:** Pandas, NumPy, Scikit-learn

---

## ⚙️ Instalação e Execução (Meta3)

1.  **Clonar o repositório:**
    ```bash
    git clone [https://github.com/lucaspagnano/Projeto-Machine-Learning-Deep-Learning.git](https://github.com/lucaspagnano/Projeto-Machine-Learning-Deep-Learning.git)
    cd Projeto-Machine-Learning-Deep-Learning/Meta3
    ```

2.  **Instalar dependências:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn streamlit SwarmPackagePy matplotlib seaborn openpyxl
    ```

3.  **Executar a Aplicação Web:**
    ```bash
    streamlit run app.py
    ```

4.  **Treinar/Otimizar (Opcional):**
    * Para correr a otimização GWO: `python otimizar_transfer.py`
    * Para gerar os modelos finais: `python treino.py`

---

## 📂 Estrutura do Projeto

* `app.py`: Aplicação Web (Streamlit) para demonstração.
* `setup_transfer.py`: Definição da arquitetura da rede (MobileNetV2 + Top Layers).
* `otimizar_transfer.py`: Script de otimização com GWO e Random Search.
* `treino.py`: Script de treino final e validação de redução de dados (100%, 50%, 25%).
* `teste.py`: Script para geração da Matriz de Confusão e Relatório de Classificação.

---
*Projeto realizado no âmbito da Unidade Curricular de Inteligência Computacional - Politécnico de Coimbra.*
