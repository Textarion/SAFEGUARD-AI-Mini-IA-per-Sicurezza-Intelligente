# domanda 12
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 1. LabelEncoder
le = LabelEncoder()
le.fit(ordine)

# 2. Target
y = le.transform(df['evento'])

# 3. Features
X = df[['ora_evento', 'movimento', 'suono_db', 'n_sagome', 'zona_vietata']]

# 4. Train/Test split (con stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#domanda 13
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Modello
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

# 2. Training
model.fit(X_train, y_train)

# 3. Predizioni
y_pred = model.predict(X_test)

# 4. Valutazione
acc = accuracy_score(y_test, y_pred)
print("Accuratezza:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=ordine))

# parte6 domanda 14
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
fig.suptitle("Matrice di Confusione", fontsize=14)

im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(5))
ax.set_yticks(range(5))

ax.set_xticklabels(ordine, rotation=25, fontsize=7)
ax.set_yticklabels(ordine, fontsize=7)

# Scrittura valori
for i in range(5):
    for j in range(5):
        color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
        ax.text(j, i, cm[i, j],
                ha="center", va="center",
                color=color, fontsize=11)

ax.set_xlabel("Predetto")
ax.set_ylabel("Reale")

plt.tight_layout()
plt.show()

#parte15
import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
fig.suptitle("Importanza delle Feature", fontsize=14)

feature_names = [
    'Ora evento',
    'Intensità\nmovimento',
    'Suono (dB)',
    'N. sagome',
    'Zona\nvietata'
]

importanze = pd.Series(model.feature_importances_, index=feature_names)
importanze = importanze.sort_values()

max_val = importanze.max()

colors = [
    '#e74c3c' if v > max_val * 0.8 else '#3498db'
    for v in importanze
]

ax.barh(importanze.index, importanze.values, color=colors)

# Valori accanto alle barre
for i, v in enumerate(importanze.values):
    ax.text(v, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("valle_templi_risultati.png", dpi=300)
plt.show()

