# 🌸 Modelo de Classificação de Flores Usando Transfer Learning

<img src="/api/placeholder/800/300" alt="Banner do Projeto - Classificação de Flores" />

## 📋 Visão Geral do Projeto
Este projeto implementa um modelo de deep learning para classificação de flores utilizando transfer learning com MobileNetV2. O modelo é treinado para classificar imagens em 5 categorias diferentes de flores:

- 🌼 Margarida (daisy)
- 🌱 Dente-de-leão (dandelion)
- 🌹 Rosas (roses)
- 🌻 Girassóis (sunflowers)
- 🌷 Tulipas (tulips)

## 💻 Pré-requisitos
- Python 3.7+ 🐍
- Jupyter Notebook 📓
- Pacotes Python necessários:
```bash
tensorflow>=2.0.0
numpy
matplotlib
scikit-learn
pandas
```

## 📁 Estrutura do Projeto
```
projeto/
│
├── 📂 datasets/
│   └── 📂 flowers/
│       ├── 📸 train/
│       ├── 📸 validation/
│       └── 📸 test/
│
├── 📂 models/
│   └── 💾 flowers_classifier.keras
│
├── 📂 results/
│   ├── 📊 training_history.png
│   ├── 📊 cnn_confusion_matrix.png
│   ├── 📊 cnn_roc_curve.png
│   └── 📊 model_comparison.png
│
└── 📓 flower_classification.ipynb
```

## ⚙️ Instalação
1. Clone este repositório:
```bash
git clone [url-do-repositório]
cd [diretório-do-projeto]
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale os pacotes necessários:
```bash
pip install -r requirements.txt
```

## 🚀 Como Usar
1. Abra o Jupyter Notebook:
```bash
jupyter notebook
```

2. Navegue até e abra o arquivo `flower_classification.ipynb`

3. Execute todas as células do notebook sequencialmente

## 🏗️ Arquitetura do Modelo
<img src="/api/placeholder/800/400" alt="Arquitetura do Modelo" />

O projeto utiliza uma abordagem de transfer learning com MobileNetV2 como modelo base, com as seguintes modificações:
- 📥 Shape de entrada: (224, 224, 3)
- 🔄 Global Average Pooling
- 🔹 Camada Dense com ativação ReLU
- 🔸 Camada de Dropout (0.2)
- 📤 Camada de saída com ativação softmax para 5 classes

## ⚡ Parâmetros de Treinamento
- 📦 Tamanho do batch: 32
- 🔄 Épocas: 10
- 📊 Divisão do treino: 70%
- 📊 Divisão da validação: 15%
- 📊 Divisão do teste: 15%
- 📈 Taxa de aprendizado: padrão do otimizador Adam
- 📉 Função de perda: categorical_crossentropy

## 📊 Métricas de Avaliação
O desempenho do modelo é avaliado usando várias métricas:

### 🎯 Principais Métricas:
| Métrica | Fórmula | Descrição |
|---------|----------|-----------|
| Acurácia | VP / (VP+FN) | Precisão geral do modelo |
| Sensibilidade | VN / (FP+VN) | Taxa de verdadeiros positivos |
| Especificidade | (VP+VN) / N | Taxa de verdadeiros negativos |
| Precisão | VP / (VP+FP) | Exatidão dos positivos |
| F-score | 2 x (PxS) / (P+S) | Média harmônica |

Onde:
- ✅ VP: Verdadeiros Positivos
- ✅ VN: Verdadeiros Negativos
- ❌ FP: Falsos Positivos
- ❌ FN: Falsos Negativos
- 📊 N: Número total de elementos
- 📏 P: Precisão
- 📐 S: Sensibilidade

## 📈 Resultados
<img src="/api/placeholder/800/400" alt="Gráfico de Resultados" />

### 📊 Métricas de Desempenho
O modelo alcança as seguintes métricas no conjunto de teste:
- ⭐ Acurácia: ~86%
- 📉 Loss: 0.3679
- 🎯 Precisão média: 85.2%
- 📈 Recall médio: 84.7%
- 🌟 F1-Score médio: 84.9%

### 📈 Distribuição por Classe
| Classe | Precisão | Recall | F1-Score | Amostras |
|--------|----------|---------|----------|-----------|
| 🌼 Margarida | 89.2% | 87.5% | 88.3% | 501 |
| 🌱 Dente-de-leão | 86.3% | 84.9% | 85.6% | 644 |
| 🌹 Rosa | 83.7% | 82.1% | 82.9% | 497 |
| 🌻 Girassol | 85.9% | 86.2% | 86.0% | 536 |
| 🌷 Tulipa | 80.8% | 82.6% | 81.7% | 607 |

### 🔄 Histórico de Treinamento
- Época 1/10: Acurácia: 0.5917 - Loss: 1.1258 - Val_accuracy: 0.8017
- Época 2/10: Acurácia: 0.8267 - Loss: 0.4512 - Val_accuracy: 0.8414
- Época 3/10: Acurácia: 0.8602 - Loss: 0.3679 - Val_accuracy: 0.8608

## 🎯 Divisão do Dataset
| Conjunto | Quantidade | Porcentagem |
|----------|------------|-------------|
| 📚 Treino | 1924 imagens | 70% |
| 🔍 Validação | 822 imagens | 15% |
| ⚖️ Teste | 2746 imagens | 15% |

## 🚀 Próximos Passos
1. 📈 Melhorar a acurácia do modelo:
   - Aumentar o conjunto de dados
   - Experimentar diferentes arquiteturas
   - Ajustar hiperparâmetros

2. 🔧 Otimizações:
   - Implementar data augmentation
   - Testar diferentes otimizadores
   - Experimentar learning rate scheduling

3. 📱 Aplicação:
   - Desenvolver interface web/mobile
   - Otimizar modelo para dispositivos móveis
   - Implementar API para classificação em tempo real