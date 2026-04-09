# SafeGuard AI 🛡️
### Sistema di Videosorveglianza Intelligente — Valle dei Templi, Agrigento

SafeGuard AI è un sistema di sorveglianza basato su machine learning che analizza i dati provenienti da sensori ambientali per classificare automaticamente gli eventi rilevati in un sito archeologico, distinguendo visitatori regolari, fauna, intrusioni e atti vandalici.

---

## Indice

- [Descrizione](#descrizione)
- [Struttura del progetto](#struttura-del-progetto)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Dataset sintetico](#dataset-sintetico)
- [Classi di eventi](#classi-di-eventi)
- [Modello ML](#modello-ml)
- [Output generati](#output-generati)
- [Simulatore eventi](#simulatore-eventi)

---

## Descrizione

Il sistema elabora 5 segnali sensoriali in tempo reale:

| Feature | Descrizione | Range |
|---|---|---|
| `ora_evento` | Ora del giorno | 0 – 24 |
| `movimento` | Intensità del movimento rilevato | 0 – 100 |
| `suono_db` | Livello sonoro in decibel | 0 – 120 |
| `n_sagome` | Numero di sagome umane/animali rilevate | 0 – 10 |
| `zona_vietata` | L'evento si trova in un'area riservata | 0 / 1 |

Sulla base di questi segnali, il sistema classifica ogni evento in una delle 5 categorie e attiva il protocollo di risposta appropriato.

---

## Struttura del progetto

```
safeguard_ai/
│
├── safeguard_ai.py          # Script principale (unico file)
├── valle_templi_eda.png     # Grafici analisi esplorativa (generato)
├── valle_templi_risultati.png  # Matrice di confusione + importanza feature (generato)
└── README.md
```

---

## Requisiti

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn

---

## Installazione

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## Utilizzo

```bash
python safeguard_ai.py
```

Lo script esegue in sequenza tutte le fasi del sistema, dall'analisi dei dati alla simulazione finale.

---

## Dataset sintetico

Il dataset viene generato proceduralmente con `numpy` a partire da range realistici per ciascuna classe. Sono prodotte **1000 osservazioni** distribuite tra 5 categorie, con un bias notturno applicato automaticamente alle classi INTRUSIONE e VANDALISMO (65% degli eventi tra le 21:00 e le 07:00).

Ogni riga include un timestamp progressivo con frequenza di 45 minuti, a partire dal `2025-06-01`.

---

## Classi di eventi

| Classe | Descrizione | Colore |
|---|---|---|
| `NORMALE` | Attività ambientale ordinaria (vento, piccoli movimenti) | ⬜ Grigio |
| `FAUNA` | Animali o fauna che attraversano l'area | 🟡 Arancio |
| `TURISTA_OK` | Visitatori nelle aree consentite durante l'orario di apertura | 🟢 Verde |
| `INTRUSIONE` | Presenza umana non autorizzata, spesso notturna | 🟠 Arancio scuro |
| `VANDALISMO` | Atto vandalico: movimenti intensi, suono elevato, zona vietata | 🔴 Rosso |

---

## Modello ML

Il classificatore utilizzato è un **Random Forest** di scikit-learn.

```
n_estimators : 100
max_depth    : 8
test_size    : 20%
stratify     : sì (bilancia le classi nel train/test split)
```

La valutazione include:
- **Accuracy score** sul test set
- **Classification report** per classe (precision, recall, F1)
- **Matrice di confusione** visualizzata come heatmap
- **Importanza delle feature** ordinata per valore (barre orizzontali)

---

## Output generati

### `valle_templi_eda.png`
Tre grafici affiancati:
1. Distribuzione oraria degli eventi critici per fascia (notte, alba, apertura, chiusura)
2. Numero medio di sagome rilevate per classe
3. Percentuale di eventi in zona vietata per classe

### `valle_templi_risultati.png`
Due grafici:
1. Matrice di confusione del modello sul test set
2. Importanza delle 5 feature nel processo decisionale del Random Forest

---

## Simulatore eventi

La parte finale del sistema simula 5 nuovi eventi realistici e per ciascuno:
- esegue la classificazione con il modello addestrato
- mostra la classe predetta con la percentuale di confidenza
- attiva il protocollo di risposta corrispondente

### Protocolli di risposta

| Classe | Protocollo |
|---|---|
| NORMALE | 🟢 Nessuna azione. Log automatico. |
| FAUNA | 🟡 Annotato nel registro fauna. Nessun allarme. |
| TURISTA_OK | 🔵 Visitatore regolare. Buona visita! |
| INTRUSIONE | 🔴 ALLERTA — Notifica Carabinieri + Guardia sito. |
| VANDALISMO | 🚨 EMERGENZA — Carabinieri + Soprintendenza + Videoregistrazione. |

Gli eventi classificati come INTRUSIONE o VANDALISMO vengono registrati in un **registro allerte** con timestamp, tipo, confidenza, zona e numero di sagome rilevate.

---

## Progetto scolastico

Sviluppato nell'ambito del progetto **Smart Valley Monitor** — I.I.S. Enrico Fermi, Aragona (AG).