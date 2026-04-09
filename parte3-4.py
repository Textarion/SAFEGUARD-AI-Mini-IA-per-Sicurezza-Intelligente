# --- PARTE 3: Registro Intrusioni ---

# Domanda 6: filtro eventi critici + conversione ora
eventi_critici = df[df['evento'].isin(['INTRUSIONE', 'VANDALISMO'])].copy()

eventi_critici['ora_str'] = eventi_critici['ora'].apply(
    lambda h: f"{int(h):02d}:{int((h % 1)*60):02d}"
)

# Domanda 7: stampa ultimi 10 eventi + statistiche

# Intestazione tabella
print(f"{'Timestamp':<18} {'Ora':<6} {'Tipo':<14} {'Sagome':<7} {'Zona vietata'}")
print("-" * 60)

# Ultimi 10 eventi critici
for _, row in eventi_critici.tail(10).iterrows():
    zona = "SI" if row['zona_vietata'] == 1 else "no"
    print(f" {str(row['timestamp'])[:16]:<18} {row['ora_str']:<6} {row['evento']:<14} {row['n_sagome']:.1f} {zona}")

# Statistiche finali
tot = len(eventi_critici)
intrusioni = (eventi_critici['evento'] == 'INTRUSIONE').sum()
vandalismi = (eventi_critici['evento'] == 'VANDALISMO').sum()
in_zona = int(eventi_critici['zona_vietata'].sum())

print("\n--- STATISTICHE ---")
print(f"Totale eventi critici: {tot}")
print(f"Intrusioni: {intrusioni}")
print(f"Vandalismi: {vandalismi}")
print(f"In zona vietata: {in_zona}")

# --- PARTE 4: Grafico EDA ---

import matplotlib.pyplot as plt

# Domanda 8: setup
ordine = ['NORMALE', 'FAUNA', 'TURISTA_OK', 'INTRUSIONE', 'VANDALISMO']
colori = ['#95a5a6', '#f39c12', '#2ecc71', '#e67e22', '#e74c3c']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('SAFEGUARD AI — Valle dei Templi: Analisi Sensori')

# Domanda 9: distribuzione oraria eventi critici
fasce = [0, 6, 8, 19, 21, 24]
fasce_label = ['Notte\n(0-6)', 'Alba\n(6-8)', 'Apertura\n(8-19)', 'Chiusura\n(19-21)', 'Notte\n(21-24)']

for tipo, colore in zip(['INTRUSIONE', 'VANDALISMO'], ['#e67e22', '#e74c3c']):
    ore = eventi_critici[eventi_critici['evento'] == tipo]['ora']
    
    conteggi = [
        ((ore >= fasce[i]) & (ore < fasce[i+1])).sum()
        for i in range(len(fasce) - 1)
    ]
    
    axes[0].bar(fasce_label, conteggi, color=colore, alpha=0.6,
                edgecolor='black', linewidth=0.6, label=tipo)

axes[0].set_title('Distribuzione Oraria\nEventi Critici')
axes[0].set_ylabel('Numero eventi')
axes[0].legend()
axes[0].tick_params(axis='x', labelsize=7)

# Domanda 10: sagome medie
medie_sagome = [
    df[df['evento'] == c]['n_sagome'].mean()
    for c in ordine
]

axes[1].bar(ordine, medie_sagome, color=colori)

for i, v in enumerate(medie_sagome):
    axes[1].text(i, v + 0.05, f'{v:.1f}', ha='center', fontsize=9)

axes[1].set_title('Sagome Medie per Evento')
axes[1].set_ylabel('Numero medio sagome')
axes[1].tick_params(axis='x', rotation=20, labelsize=8)

# Domanda 11: percentuale zona vietata
prop_vietata = [
    df[df['evento'] == c]['zona_vietata'].mean() * 100
    for c in ordine
]

axes[2].bar(ordine, prop_vietata, color=colori)
axes[2].set_ylim(0, 105)

axes[2].axhline(50, color='red', linestyle='--', linewidth=1,
                alpha=0.6, label='50%')

for i, v in enumerate(prop_vietata):
    axes[2].text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=9)

axes[2].set_title('Eventi in Zona Vietata (%)')
axes[2].legend()
axes[2].tick_params(axis='x', rotation=20, labelsize=8)

# Salvataggio e visualizzazione
plt.tight_layout()
plt.savefig('valle_templi_eda.png', dpi=120)
plt.show()