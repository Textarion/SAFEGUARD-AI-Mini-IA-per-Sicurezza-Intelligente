# SafeGuard AI è un sistema di videosorveglianza alimentato ad intelligenza artificiale, scritto con python e con l'aiuto di diverse
# librerie aiuta ad identificare le masse con potenziali vandalismi e intrusioni, garantendo la sicurezza 
# di un posto archeologico come la valle dei templi.


# Importazione librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import datetime as dt

# Setto il seme per randomizzare
np.random.seed(13)

# Evento esempio
ora_evento = 14          # ora del giorno (0-24)
movimento = 65           # intensità movimento (0-100)
suono_db = 40            # livello sonoro in decibel (0-120)
n_sagome = 2             # numero di sagome rilevate (0-10)
zona_vietata = 1         # 1 = area non accessibile, 0 = area normale

evento1 = np.array([ora_evento, movimento, suono_db, n_sagome, zona_vietata])

# Dizionario delle classi di eventi con i range dei sensori
classi = {

    # Attività normale del sito durante il giorno (vento, piccoli movimenti ambientali)
    'NORMALE': {
        'ora_evento': (6, 22),
        'movimento': (2, 20),
        'suono_db': (5, 35),
        'n_sagome': (0, 0.4),
        'zona_vietata': (0, 0.05),
        'n': 300
    },

    # Animali o fauna che attraversano l'area
    'FAUNA': {
        'ora_evento': (0, 24),
        'movimento': (15, 55),
        'suono_db': (10, 50),
        'n_sagome': (0, 0.8),
        'zona_vietata': (0, 0.3),
        'n': 200
    },

    # Turisti presenti nelle aree consentite durante l'orario di visita
    'TURISTA_OK': {
        'ora_evento': (8, 19),
        'movimento': (30, 75),
        'suono_db': (35, 75),
        'n_sagome': (1, 8),
        'zona_vietata': (0, 0.05),
        'n': 250
    },

    # Intrusione notturna in zone vietate
    'INTRUSIONE': {
        'ora_evento': (0, 24),
        'movimento': (40, 90),
        'suono_db': (20, 70),
        'n_sagome': (1, 4),
        'zona_vietata': (0.7, 1.0),
        'n': 160
    },

    # Atto vandalico: rumore alto, movimenti intensi e presenza umana
    'VANDALISMO': {
        'ora_evento': (20, 24),
        'movimento': (50, 100),
        'suono_db': (55, 110),
        'n_sagome': (1, 5),
        'zona_vietata': (0.5, 1.0),
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

        'ora_evento': ora,

        'movimento': np.random.uniform(*p['movimento'], nn),

        'suono_db': np.random.uniform(*p['suono_db'], nn),

        'n_sagome': np.random.uniform(*p['n_sagome'], nn).clip(0, 10),

        'zona_vietata': np.random.uniform(*p['zona_vietata'], nn).clip(0, 1).round(),

        'evento': etich
    })

    frames.append(df_temp)

    df = pd.concat(frames)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.insert(
    0,
    'timestamp',
    pd.date_range('2025-06-01', periods=len(df), freq='45min')
)

