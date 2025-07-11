import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import shap
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline

# === 1. Gerar 100 tweets simulados ===
positivos = [
    "Amei o produto", "Excelente atendimento", "Muito satisfeito",
    "Serviço maravilhoso", "Recomendo com certeza", "Nota 10!",
    "Funcionou perfeitamente", "Tudo ótimo", "Parabéns à equipe",
    "Voltarei a comprar", "Compra fácil e rápida", "Entrega no prazo",
    "Produto de qualidade", "Atendimento incrível", "Super recomendo",
    "Perfeito!", "Muito bom mesmo", "Superou minhas expectativas",
    "Simplesmente sensacional", "Fantástico"
]

negativos = [
    "Péssimo atendimento", "Não funcionou", "Horrível experiência",
    "Nunca mais compro", "Produto quebrado", "Entrega atrasada",
    "O pior serviço", "Muito ruim", "Experiência negativa",
    "Totalmente insatisfeito", "Decepcionado", "Não recomendo",
    "Demorou demais", "Suporte horrível", "Problemas constantes",
    "Nada bom", "Desastre total", "Fiquei frustrado",
    "Sem qualidade", "Inútil"
]

# Repetir aleatoriamente para totalizar 100
tweets = random.choices(positivos, k=50) + random.choices(negativos, k=50)
labels = [1]*50 + [0]*50

# === 2. Vetorização ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweets)
y = np.array(labels)

# === 3. Separar treino e teste ===
X_train, X_test, y_train, y_test, tweets_train, tweets_test = train_test_split(
    X, y, tweets, test_size=0.2, stratify=y, random_state=42
)

# === 4. Modelo ===
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# === 5. Acurácia ===
preds = model.predict(X_test)
print("Acurácia:", accuracy_score(y_test, preds))

# === 6. Preparar explicadores ===
class_names = ['negativo', 'positivo']
pipeline = make_pipeline(vectorizer, model)
explainer_lime = lime.lime_text.LimeTextExplainer(class_names=class_names)
explainer_shap = shap.Explainer(model, X_train.toarray())

# === 7. Comparação ===
lime_times = []
shap_times = []
lime_mse = []
shap_mse = []
tweets_analisados = []

# Analisar apenas as 10 primeiras amostras de teste
for i, tweet in enumerate(tweets_test[:10]):
    print(f"\nAnalisando: \"{tweet}\"")
    tweets_analisados.append(tweet)

    # LIME
    start = time.time()
    exp = explainer_lime.explain_instance(tweet, pipeline.predict_proba, num_features=6)
    lime_time = time.time() - start
    lime_pred = exp.predict_proba[1]
    model_pred = pipeline.predict_proba([tweet])[0][1]
    lime_error = (model_pred - lime_pred) ** 2
    lime_times.append(lime_time)
    lime_mse.append(lime_error)

    # SHAP
    start = time.time()
    X_tweet = vectorizer.transform([tweet]).toarray()
    shap_vals = explainer_shap(X_tweet)
    shap_time = time.time() - start
    shap_sum = shap_vals.values[0].sum() + shap_vals.base_values[0]
    shap_error = (model_pred - shap_sum) ** 2
    shap_times.append(shap_time)
    shap_mse.append(shap_error)

# === 8. Resultados médios ===
media_lime_time = np.mean(lime_times)
media_shap_time = np.mean(shap_times)
media_lime_mse = np.mean(lime_mse)
media_shap_mse = np.mean(shap_mse)

print("\n==== Comparação Quantitativa ====")
print(f"Tempo médio LIME: {media_lime_time:.4f} s")
print(f"Tempo médio SHAP: {media_shap_time:.4f} s")
print(f"Fidelidade local (MSE) LIME: {media_lime_mse:.6f}")
print(f"Fidelidade local (MSE) SHAP: {media_shap_mse:.6f}")

# === 9. Gráficos ===
x = np.arange(len(tweets_analisados))
width = 0.35

# Tempo por amostra
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, lime_times, width, label='LIME', color='orange')
plt.bar(x + width/2, shap_times, width, label='SHAP', color='blue')
plt.xticks(x, [f"Amostra {i+1}" for i in x], rotation=45)
plt.ylabel("Tempo (s)")
plt.title("Tempo de execução por amostra")
plt.legend()
plt.tight_layout()
plt.show()

# Fidelidade por amostra
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, lime_mse, width, label='LIME', color='orange')
plt.bar(x + width/2, shap_mse, width, label='SHAP', color='blue')
plt.xticks(x, [f"Amostra {i+1}" for i in x], rotation=45)
plt.ylabel("Erro quadrático (MSE)")
plt.title("Fidelidade local por amostra")
plt.legend()
plt.tight_layout()
plt.show()

# Médias comparativas
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].bar(["LIME", "SHAP"], [media_lime_time, media_shap_time], color=["orange", "blue"])
axs[0].set_title("Tempo médio de execução")
axs[0].set_ylabel("Tempo (s)")

axs[1].bar(["LIME", "SHAP"], [media_lime_mse, media_shap_mse], color=["orange", "blue"])
axs[1].set_title("Fidelidade local média (MSE)")
axs[1].set_ylabel("Erro quadrático")

plt.tight_layout()
plt.show()
