"""
explorative Datenanalyse
"""
# Vorbereitung

# Biblipotheken importieren
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Produktionsdaten einlesen
data = pd.read_excel('C:\Datensaetze generieren/PSP_Jan_Feb_2019.xlsx')

# Spalten formatieren
data['tmsp'] = data['tmsp'].astype(str)

# Datentypen ausgeben
print(data.dtypes)

"""
Merkamale generieren und im Dataframe hinzufügen
"""
# Aufsplitten der Spalte tmsp in Jahr, Montag, Tag, Uhrzeit und Datum
data[['Jahr','Monat','Tag']] = data['tmsp'].str.split('-', n = 3, expand =True)
data['Tag'] = data['Tag'].str.slice(stop=-9)
data['uhrzeit'] = data['tmsp'].str.slice(start=11)

# Servicegebuehren der PSPs hinzufügen
data['gebuehr'] = 0
data.loc[(data['PSP'] == 'Moneycard') & (data['success'] == 1), 'gebuehr'] = 5
data.loc[(data['PSP'] == 'Moneycard') & (data['success'] == 0), 'gebuehr'] = 2
data.loc[(data['PSP'] == 'Goldcard') & (data['success'] == 1), 'gebuehr'] = 10
data.loc[(data['PSP'] == 'Goldcard') & (data['success'] == 0), 'gebuehr'] = 5
data.loc[(data['PSP'] == 'UK_Card') & (data['success'] == 1), 'gebuehr'] = 3
data.loc[(data['PSP'] == 'UK_Card') & (data['success'] == 0), 'gebuehr'] = 1
data.loc[(data['PSP'] == 'Simplecard') & (data['success'] == 1), 'gebuehr'] = 1
data.loc[(data['PSP'] == 'Simplecard') & (data['success'] == 0), 'gebuehr'] = 0.5


# DataFrame als Excel-Datei speichern
#data.to_excel('C:\Datensaetze generieren/datei.xlsx', index=False)

"""
weitere Variablen für die explorative Datenanalyse generieren
"""
# Erfolge pro PSP
erfolg_moneycard = len(data[(data['PSP'] == 'Moneycard') & (data['success'] == 1)])
erfolg_goldcard = len(data[(data['PSP'] == 'Goldcard') & (data['success'] == 1)])
erfolg_ukcard = len(data[(data['PSP'] == 'UK_Card') & (data['success'] == 1)])
erfolg_simplecard = len(data[(data['PSP'] == 'Simplecard') & (data['success'] == 1)])

# Erfolge pro card
erfolg_master = len(data[(data['card'] == 'Master') & (data['success'] == 1)])
erfolg_diners = len(data[(data['card'] == 'Diners') & (data['success'] == 1)])
erfolg_visa = len(data[(data['card'] == 'Visa') & (data['success'] == 1)])

# Konfidenzintervalle
# Aggregation nach Stunden/ Tag / Monat und Berechnung der Anzahl der Transaktionen pro Stunde / Tag / Monat
data['Hour'] = pd.to_datetime(data['uhrzeit']).dt.hour
hourly_counts = data.groupby('Hour').size()
day_counts = data.groupby('Tag').size()
month_counts = data.groupby('Monat').size()

# Berechnung des Konfidenzintervalls für die Anzahl der Transaktionen pro Stunde
mean_counts = hourly_counts.mean()
std_counts = hourly_counts.std()
confidence_interval = 1.96 * (std_counts / np.sqrt(len(hourly_counts)))

# Berechnung des Konfidenzintervalls für die Anzahl der Transaktionen pro Tag
mean_counts_d = day_counts.mean()
std_counts_d = day_counts.std()
confidence_interval_t = 1.96 * (std_counts_d / np.sqrt(len(day_counts)))


# Anzahl der Erfolge nach PSP gruppieren
success_psp = data[data['success'] == 1].groupby('PSP').size()
# Anzahl der Erfolge nach Kartenanbieter gruppieren
success_card = data[data['success'] == 1].groupby('card').size()
# Anzahl der Erfolge nach Gebuehr gruppieren
success_gebuehr = data[data['success'] == 1].groupby('gebuehr').size()
# Anzahl der Erfolge nach Gebuehr gruppieren
unsuccess_gebuehr = data[data['success'] == 0].groupby('gebuehr').size()
# Anzahl der Erfolge nach Gebuehr gruppieren
Dsecured_ja = data[data['3D_secured'] == 1].groupby('success').size()
# Anzahl der Erfolge nach Gebuehr gruppieren
Dsecured_nein = data[data['3D_secured'] == 0].groupby('success').size()

# Filtern nach dem Land
germany_transactions = data[data['country'] == 'Germany']['amount']
austria_transactions = data[data['country'] == 'Austria']['amount']
switzerland_transactions = data[data['country'] == 'Switzerland']['amount']

# Korrelationsmatrix berechnen
correlation_matrix = data.corr()

# durchschnittliche Gebühr berechnen
gebuehr_avg = data.groupby(['card', 'PSP']).mean()['gebuehr']
# durchschnittliche Erfolgsrate berechnen
erfolgsrate_avg = data.groupby(['card', 'PSP']).mean()['success']


"""
Grafiken erstellen
"""

# Balkendiagramm erstellen
plt.figure(num='PSP')
plt.bar(success_psp.index, success_psp.values, width=0.3, edgecolor='black', linewidth=1.2)
plt.xlabel('PSP')
plt.ylabel('Anzahl')
plt.title("erfolgreiche Transaktionen je PSP")

plt.figure(num='card')
plt.bar(success_card.index, success_card.values, width=0.3, edgecolor='black', linewidth=1.2)
plt.xlabel('Kartenanbieter')
plt.ylabel('Anzahl')
plt.title("erfolgreiche Transaktionen je Karte")

# X-Koordinaten der Balken
x = np.arange(2)
bar_width = 0.25
plt.figure(num='secured')
plt.bar(x, Dsecured_ja.values / Dsecured_ja.sum() * 100, width=bar_width, align='edge', edgecolor='black', linewidth=1.2, label = '3D_secure_ja')
plt.bar(x + bar_width, Dsecured_nein.values / Dsecured_nein.sum() * 100, width=bar_width, align='edge', edgecolor='black', linewidth=1.2, label = '3D_secure_nein')
plt.xlabel('success')
plt.ylabel('Verteilung in Prozent')
plt.title("Transaktionen mit 3D_secured")
plt.legend()
# x-Achsenbeschriftungen festlegen
plt.xticks([0.25,1.25],['gescheiterte Transaktionen','erfolgreiche Transaktionen'])

# Histogramm erstellen
plt.figure(num='amount')
plt.subplot(2, 2, 1) 
plt.hist(data['amount'], bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
plt.xlabel('Transaktionsbetrag gesamt')
plt.ylabel('Anzahl')
plt.subplot(2, 2, 2) 
plt.hist(germany_transactions, bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
plt.xlabel('Transaktionsbetrag Deutschland')
plt.ylabel('Anzahl')
plt.subplot(2, 2, 3) 
plt.hist(austria_transactions, bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
plt.xlabel('Transaktionsbetrag Australien')
plt.ylabel('Anzahl')
plt.subplot(2, 2, 4) 
plt.hist(switzerland_transactions, bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
plt.xlabel('Transaktionsbetrag Schweiz')
plt.ylabel('Anzahl')
# Abstand zwischen den Subplots anpassen
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Erstellen eines Zeitreihenplots mit Konfidenzintervall
plt.figure(num='konfi', figsize=(8, 6))  # Figure-Größe anpassen
plt.subplot(3,1,1)
plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-', label='Transaktionen pro Stunde')
plt.fill_between(hourly_counts.index, mean_counts-confidence_interval, mean_counts+confidence_interval, color='gray', alpha=0.3, label='Konfidenzintervall')
# Achsentitel hinzufügen
plt.xlabel('Stunde')
plt.ylabel('Anzahl der Transaktionen')
# Legende anzeigen
plt.legend()
plt.subplot(3,1,2)
plt.plot(day_counts.index, day_counts.values, marker='o', linestyle='-', label='Transaktionen pro Tag')
plt.fill_between(day_counts.index, mean_counts_d-confidence_interval_t, mean_counts_d+confidence_interval_t, color='gray', alpha=0.3, label='Konfidenzintervall')
# Achsentitel hinzufügen
plt.xlabel('Tag')
plt.ylabel('Anzahl der Transaktionen')
# Legende anzeigen
plt.legend()
plt.subplot(3,1,3)
plt.plot(month_counts.index, month_counts.values, marker='o', linestyle='-', label='Transaktionen pro Monat')
# Achsentitel hinzufügen
plt.xlabel('Monat')
plt.ylabel('Anzahl der Transaktionen')
# Legende anzeigen
plt.legend()
plt.subplots_adjust(hspace=0.4, wspace=0.1)

# Boxplot erstellen
plt.figure(num='Transaktionen pro Land', figsize=(8, 6))
plt.subplot(3, 1, 1) 
plt.boxplot([data[data['country'] == 'Germany']['amount'],
             data[data['country'] == 'Austria']['amount'],
             data[data['country'] == 'Switzerland']['amount']])
# Achsentitel hinzufügen
plt.ylabel('Transaktionsbetrag')
# Länderbeschriftungen hinzufügen
plt.xticks([1, 2, 3], ['Deutschland', 'Australien', 'Schweiz'])
plt.subplot(3, 1, 2) 
plt.boxplot([data[data['card'] == 'Diners']['amount'],
             data[data['card'] == 'Master']['amount'],
             data[data['card'] == 'Visa']['amount']])
# Achsentitel hinzufügen
plt.ylabel('Transaktionsbetrag')
# Länderbeschriftungen hinzufügen
plt.xticks([1, 2, 3], ['Diners', 'Master', 'Visa'])
plt.subplot(3, 1, 3) 
plt.boxplot([data[data['PSP'] == 'Goldcard']['amount'],
             data[data['PSP'] == 'Moneycard']['amount'],
             data[data['PSP'] == 'Simplecard']['amount'],
             data[data['PSP'] == 'UK_Card']['amount']])
# Achsentitel hinzufügen
plt.ylabel('Transaktionsbetrag')
# Länderbeschriftungen hinzufügen
plt.xticks([1, 2, 3, 4], ['Goldcard', 'Moneycard', 'Simplecard', 'UK_Card'])
# Abstand zwischen den Subplots anpassen
plt.subplots_adjust(hspace=0.2, wspace=0.1)

# Heatmap erstellen
plt.figure(num='Korrelation')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Durchschnittserfolgsrate anzeigen, welche Kreditkarte mit welchem Zahlungsdienstleister am erfolgreichsten war
# Gruppierung und Berechnung der Erfolgsrate
plt.figure(num='Durchschnittserfolg')
# Gestapeltes Balkendiagramm erstellen
erfolgsrate_avg.unstack().plot(kind='bar', stacked=True)
# Achsentitel hinzufügen
plt.xlabel('Kreditkarte')
plt.ylabel('Durchschnittserfolgsrate')
# Legende hinzufügen
plt.legend(title='Zahlungsdienstleister', bbox_to_anchor=(1, 1))

# Servicegebühr anzeigen, welche Kreditkarte mit welchem Zahlungsdienstleister
# Gruppierung nach Kreditkarte und Zahlungsanbieter, und Berechnung des Durchschnitts der Servicegebühr
plt.figure(num='bar_Servicegebuehr_erfolgsrate')
# Gruppierung nach Kreditkarte und Zahlungsanbieter, und Berechnung des Durchschnitts der Servicegebühr und Erfolgsrate
# Balkendiagramm erstellen
bars = plt.bar(range(len(gebuehr_avg)), gebuehr_avg.values)
# Kreditkarten- und PSP-Kombinationen als x-Achsenticklabels setzen
tick_labels = [f"{kreditkarte} - {zahlungsanbieter}" for kreditkarte, zahlungsanbieter in gebuehr_avg.index]
plt.xticks(range(len(gebuehr_avg)), tick_labels, rotation='vertical')
# Erfolgsrate als Textlabels über den Balken hinzufügen
for i, rect in enumerate(bars):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, f"{erfolgsrate_avg.values[i]:.2f}",
             ha='center', va='bottom')
# Achsentitel hinzufügen
plt.xlabel('Kreditkarte, Zahlungsanbieter')
plt.ylabel('Durchschnittliche Servicegebühr')

# Erstellen des Streudiagramms
plt.figure(num='scatter_Servicegebuehr_erfolgsrate')
fig, ax = plt.subplots(figsize=(9, 7))  # Größe des Diagramms anpassen
ax.scatter(gebuehr_avg, erfolgsrate_avg)
# Hervorhebung der Kombination mit geringsten Servicegebühren und höchster Erfolgsrate
min_gebuehr_index = gebuehr_avg.idxmin()
max_erfolgsrate_index = erfolgsrate_avg.idxmax()
ax.scatter(gebuehr_avg[min_gebuehr_index], erfolgsrate_avg[min_gebuehr_index], color='red', label='Min. Gebühren')
ax.scatter(gebuehr_avg[max_erfolgsrate_index], erfolgsrate_avg[max_erfolgsrate_index], color='green', label='Max. Erfolgsrate')
# Namen der Kreditkartenanbieter und PSPs hinzufügen
for i, (kreditkarte, psp) in enumerate(gebuehr_avg.index):
    plt.annotate(f"{kreditkarte}, {psp}", (gebuehr_avg[i], erfolgsrate_avg[i]), ha='center', va='bottom', fontsize=8)
# Achsentitel hinzufügen
plt.xlabel('Servicegebühr')
plt.ylabel('Erfolgsrate')
plt.legend()
