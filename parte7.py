## PARTE 7 — Simulatore con Registro Allerte 
protocolli = {
    'NORMALE': ('🟢', 'Nessuna azione. Log automatico.'),
    'FAUNA': ('🟡', 'Annotato nel registro fauna. Nessun allarme.'),
    'TURISTA_OK': ('🔵', 'Visitatore regolare. Buona visita!'),
    'INTRUSIONE': ('🔴', 'ALLERTA — Notifica Carabinieri + Guardia sito.'),
    'VANDALISMO': ('🚨', 'EMERGENZA — Carabinieri + Soprintendenza + Videoregistrazione.')
}

import pandas as pd

nuovi_eventi = pd.DataFrame({
    'ora_evento': [14.5, 23.8, 2.1, 10.2, 22.5],
    'movimento': [45, 78, 12, 55, 88],
    'suono_db': [60, 35, 18, 65, 95],
    'n_sagome': [3.0, 2.0, 0.1, 4.0, 2.0],
    'zona_vietata': [0, 1, 0, 0, 1]
})

descrizioni = [
    'Gruppo visitatori — Tempio della Concordia (pomeriggio)',
    'Figura solitaria — area scavi riservata (notte)',
    'Movimento basso — zona parcheggio (notte)',
    'Gruppo numeroso — Tempio di Giunone (mattina)',
    'Persona con attrezzi — colonnato Tempio di Ercole (notte)'
]

pred_classi = le.inverse_transform(modello.predict(nuovi_eventi))
prob_tutti = modello.predict_proba(nuovi_eventi)

registro_allerte = []

from datetime import datetime

for i, (desc, classe) in enumerate(zip(descrizioni, pred_classi)):
    
    emoji, azione = protocolli[classe]
    confidenza = prob_tutti[i].max() * 100

    # Timestamp da ora decimale
    ora_h = int(nuovi_eventi['ora_evento'].iloc[i])
    ora_m = int((nuovi_eventi['ora_evento'].iloc[i] % 1) * 60)
    timestamp_now = f"2025-08-15 {ora_h:02d}:{ora_m:02d}"

    # Output
    print(f"\n[{timestamp_now}]")
    print(f"Evento: {desc}")
    print(f"Classificazione: {classe} {emoji} ({confidenza:.1f}%)")
    print(f"Azione: {azione}")

    # Registrazione allerte
    if classe in ['INTRUSIONE', 'VANDALISMO']:
        registro_allerte.append({
            'timestamp': timestamp_now,
            'tipo': classe,
            'descrizione': desc[:45],
            'confidenza': f"{confidenza:.0f}%",
            'zona_vietata': "SI" if nuovi_eventi['zona_vietata'].iloc[i] == 1 else "no",
            'sagome': nuovi_eventi['n_sagome'].iloc[i]
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