# SafeGuard AI è un sistema di videosorveglianza alimentato ad intelligenza artificiale, scritto con python e con l'aiuto di diverse
# librerie aiuta ad identificare le masse con potenziali vandalismi e intrusioni, garantendo la sicurezza 
# di un posto archeologico come la valle dei templi.


# ============================================================
# IMPORTAZIONI
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from datetime import datetime


# ============================================================
# PARTE 1 — Generazione Dataset
# ============================================================

# Setto il seme per randomizzare
np.random.seed(13)

# Evento esempio
ora_evento    = 14   # ora del giorno (0-24)
movimento     = 65   # intensità movimento (0-100)
suono_db      = 40   # livello sonoro in decibel (0-120)
n_sagome      = 2    # numero di sagome rilevate (0-10)
zona_vietata  = 1    # 1 = area non accessibile, 0 = area normale

evento1 = np.array([ora_evento, movimento, suono_db, n_sagome, zona_vietata])

# Dizionario delle classi di eventi con i range dei sensori
classi = {

    # Attività normale del sito durante il giorno (vento, piccoli movimenti ambientali)
    'NORMALE': {
        'ora_evento':    (6, 22),
        'movimento':     (2, 20),
        'suono_db':      (5, 35),
        'n_sagome':      (0, 0.4),
        'zona_vietata':  (0, 0.05),
        'n': 300
    },

    # Animali o fauna che attraversano l'area
    'FAUNA': {
        'ora_evento':    (0, 24),
        'movimento':     (15, 55),
        'suono_db':      (10, 50),
        'n_sagome':      (0, 0.8),
        'zona_vietata':  (0, 0.3),
        'n': 200
    },

    # Turisti presenti nelle aree consentite durante l'orario di visita
    'TURISTA_OK': {
        'ora_evento':    (8, 19),
        'movimento':     (30, 75),
        'suono_db':      (35, 75),
        'n_sagome':      (1, 8),
        'zona_vietata':  (0, 0.05),
        'n': 250
    },

    # Intrusione notturna in zone vietate
    'INTRUSIONE': {
        'ora_evento':    (0, 24),
        'movimento':     (40, 90),
        'suono_db':      (20, 70),
        'n_sagome':      (1, 4),
        'zona_vietata':  (0.7, 1.0),
        'n': 160
    },

    # Atto vandalico: rumore alto, movimenti intensi e presenza umana
    'VANDALISMO': {
        'ora_evento':    (20, 24),
        'movimento':     (50, 100),
        'suono_db':      (55, 110),
        'n_sagome':      (1, 5),
        'zona_vietata':  (0.5, 1.0),
        'n': 90
    }
}

frames = []

for etich, p in classi.items():

    nn = p['n']
    ora = np.random.uniform(*p['ora_evento'], nn)

    # Intrusioni e vandalismi avvengono soprattutto di notte
    if etich in ['INTRUSIONE', 'VANDALISMO']:
        ora = np.where(
            np.random.random(nn) < 0.65,
            np.random.uniform(21, 24, nn),
            np.random.uniform(0, 7, nn)
        )

    df_temp = pd.DataFrame({
        'ora_evento':   ora,
        'movimento':    np.random.uniform(*p['movimento'], nn),
        'suono_db':     np.random.uniform(*p['suono_db'], nn),
        'n_sagome':     np.random.uniform(*p['n_sagome'], nn).clip(0, 10),
        'zona_vietata': np.random.uniform(*p['zona_vietata'], nn).clip(0, 1).round(),
        'evento':       etich
    })

    frames.append(df_temp)

df = pd.concat(frames)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.insert(
    0,
    'timestamp',
    pd.date_range('2025-06-01', periods=len(df), freq='45min')
)

# Ordine e colori usati nei grafici e nell'encoder
ordine = ['NORMALE', 'FAUNA', 'TURISTA_OK', 'INTRUSIONE', 'VANDALISMO']
colori = ['#95a5a6', '#f39c12', '#2ecc71', '#e67e22', '#e74c3c']


# ============================================================
# PARTE 2 — Registro Intrusioni
# ============================================================

# Filtro eventi critici + conversione ora
eventi_critici = df[df['evento'].isin(['INTRUSIONE', 'VANDALISMO'])].copy()

# FIX: la colonna si chiama 'ora_evento', non 'ora'
eventi_critici['ora_str'] = eventi_critici['ora_evento'].apply(
    lambda h: f"{int(h):02d}:{int((h % 1) * 60):02d}"
)

# Stampa ultimi 10 eventi + statistiche
print(f"{'Timestamp':<18} {'Ora':<6} {'Tipo':<14} {'Sagome':<7} {'Zona vietata'}")
print("-" * 60)

for _, row in eventi_critici.tail(10).iterrows():
    zona = "SI" if row['zona_vietata'] == 1 else "no"
    print(f" {str(row['timestamp'])[:16]:<18} {row['ora_str']:<6} {row['evento']:<14} {row['n_sagome']:.1f} {zona}")

# Statistiche finali
tot        = len(eventi_critici)
intrusioni = (eventi_critici['evento'] == 'INTRUSIONE').sum()
vandalismi = (eventi_critici['evento'] == 'VANDALISMO').sum()
in_zona    = int(eventi_critici['zona_vietata'].sum())

print("\n--- STATISTICHE ---")
print(f"Totale eventi critici: {tot}")
print(f"Intrusioni: {intrusioni}")
print(f"Vandalismi: {vandalismi}")
print(f"In zona vietata: {in_zona}")


# ============================================================
# PARTE 3 — Grafico EDA
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('SAFEGUARD AI — Valle dei Templi: Analisi Sensori')

# Distribuzione oraria eventi critici
fasce       = [0, 6, 8, 19, 21, 24]
fasce_label = ['Notte\n(0-6)', 'Alba\n(6-8)', 'Apertura\n(8-19)', 'Chiusura\n(19-21)', 'Notte\n(21-24)']

for tipo, colore in zip(['INTRUSIONE', 'VANDALISMO'], ['#e67e22', '#e74c3c']):
    # FIX: usa 'ora_evento' (nome corretto della colonna)
    ore = eventi_critici[eventi_critici['evento'] == tipo]['ora_evento']
    conteggi = [
        ((ore >= fasce[i]) & (ore < fasce[i + 1])).sum()
        for i in range(len(fasce) - 1)
    ]
    axes[0].bar(fasce_label, conteggi, color=colore, alpha=0.6,
                edgecolor='black', linewidth=0.6, label=tipo)

axes[0].set_title('Distribuzione Oraria\nEventi Critici')
axes[0].set_ylabel('Numero eventi')
axes[0].legend()
axes[0].tick_params(axis='x', labelsize=7)

# Sagome medie per classe
medie_sagome = [df[df['evento'] == c]['n_sagome'].mean() for c in ordine]

axes[1].bar(ordine, medie_sagome, color=colori)
for i, v in enumerate(medie_sagome):
    axes[1].text(i, v + 0.05, f'{v:.1f}', ha='center', fontsize=9)

axes[1].set_title('Sagome Medie per Evento')
axes[1].set_ylabel('Numero medio sagome')
axes[1].tick_params(axis='x', rotation=20, labelsize=8)

# Percentuale zona vietata per classe
prop_vietata = [df[df['evento'] == c]['zona_vietata'].mean() * 100 for c in ordine]

axes[2].bar(ordine, prop_vietata, color=colori)
axes[2].set_ylim(0, 105)
axes[2].axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.6, label='50%')
for i, v in enumerate(prop_vietata):
    axes[2].text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)

axes[2].set_title('Eventi in Zona Vietata (%)')
axes[2].legend()
axes[2].tick_params(axis='x', rotation=20, labelsize=8)

plt.tight_layout()
plt.savefig('valle_templi_eda.png', dpi=120)
plt.show()


# ============================================================
# PARTE 4 — Modello ML: Training e Valutazione
# ============================================================

# LabelEncoder e split
le = LabelEncoder()
le.fit(ordine)

y = le.transform(df['evento'])
X = df[['ora_evento', 'movimento', 'suono_db', 'n_sagome', 'zona_vietata']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Training Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# Valutazione
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print("Accuratezza:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=ordine))

# Matrice di confusione
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
fig.suptitle("Matrice di Confusione", fontsize=14)

im = ax.imshow(cm, cmap='Blues')

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(ordine, rotation=25, fontsize=7)
ax.set_yticklabels(ordine, fontsize=7)

for i in range(5):
    for j in range(5):
        color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
        ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=11)

ax.set_xlabel("Predetto")
ax.set_ylabel("Reale")

plt.tight_layout()
plt.show()

# Importanza delle feature
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
fig.suptitle("Importanza delle Feature", fontsize=14)

feature_names = [
    'Ora evento',
    'Intensità\nmovimento',
    'Suono (dB)',
    'N. sagome',
    'Zona\nvietata'
]

importanze = pd.Series(model.feature_importances_, index=feature_names).sort_values()
max_val    = importanze.max()
colors     = ['#e74c3c' if v > max_val * 0.8 else '#3498db' for v in importanze]

ax.barh(importanze.index, importanze.values, color=colors)
for i, v in enumerate(importanze.values):
    ax.text(v, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("valle_templi_risultati.png", dpi=300)
plt.show()


# ============================================================
# PARTE 5 — Simulatore con Registro Allerte
# ============================================================

protocolli = {
    'NORMALE':    ('🟢', 'Nessuna azione. Log automatico.'),
    'FAUNA':      ('🟡', 'Annotato nel registro fauna. Nessun allarme.'),
    'TURISTA_OK': ('🔵', 'Visitatore regolare. Buona visita!'),
    'INTRUSIONE': ('🔴', 'ALLERTA — Notifica Carabinieri + Guardia sito.'),
    'VANDALISMO': ('🚨', 'EMERGENZA — Carabinieri + Soprintendenza + Videoregistrazione.')
}

nuovi_eventi = pd.DataFrame({
    'ora_evento':   [14.5, 23.8, 2.1, 10.2, 22.5],
    'movimento':    [45,   78,   12,  55,   88],
    'suono_db':     [60,   35,   18,  65,   95],
    'n_sagome':     [3.0,  2.0,  0.1, 4.0,  2.0],
    'zona_vietata': [0,    1,    0,   0,    1]
})

descrizioni = [
    'Gruppo visitatori — Tempio della Concordia (pomeriggio)',
    'Figura solitaria — area scavi riservata (notte)',
    'Movimento basso — zona parcheggio (notte)',
    'Gruppo numeroso — Tempio di Giunone (mattina)',
    'Persona con attrezzi — colonnato Tempio di Ercole (notte)'
]

# FIX: usa 'model' (nome corretto della variabile)
pred_classi = le.inverse_transform(model.predict(nuovi_eventi))
prob_tutti  = model.predict_proba(nuovi_eventi)

registro_allerte = []

for i, (desc, classe) in enumerate(zip(descrizioni, pred_classi)):

    emoji, azione = protocolli[classe]
    confidenza    = prob_tutti[i].max() * 100

    ora_h         = int(nuovi_eventi['ora_evento'].iloc[i])
    ora_m         = int((nuovi_eventi['ora_evento'].iloc[i] % 1) * 60)
    timestamp_now = f"2025-08-15 {ora_h:02d}:{ora_m:02d}"

    print(f"\n[{timestamp_now}]")
    print(f"Evento: {desc}")
    print(f"Classificazione: {classe} {emoji} ({confidenza:.1f}%)")
    print(f"Azione: {azione}")

    if classe in ['INTRUSIONE', 'VANDALISMO']:
        registro_allerte.append({
            'timestamp':    timestamp_now,
            'tipo':         classe,
            'descrizione':  desc[:45],
            'confidenza':   f"{confidenza:.0f}%",
            'zona_vietata': "SI" if nuovi_eventi['zona_vietata'].iloc[i] == 1 else "no",
            'sagome':       nuovi_eventi['n_sagome'].iloc[i]
        })

if registro_allerte:
    print("\n==============================")
    print(f"EVENTI CRITICI RILEVATI: {len(registro_allerte)}")

    reg_df = pd.DataFrame(registro_allerte)
    print("\nRegistro allerte:")
    print(reg_df.to_string(index=False))

    nome_file = f"log_allerte_{datetime.now().strftime('%Y%m%d')}.txt"
    print(f"\nLog salvabile come: {nome_file}")
else:
    print("\nNessun evento critico registrato in questa sessione.")

print("\n--- Fine simulazione sistema di sorveglianza ---")