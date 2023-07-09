"""
Created on Sat Jul  8 12:51:20 2023

explorative Datenanalyse
"""

# Pakete importieren
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

# Merkamale generieren
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

# Erfolge pro PSP
erfolg_moneycard = len(data[(data['PSP'] == 'Moneycard') & (data['success'] == 1)])
erfolg_goldcard = len(data[(data['PSP'] == 'Goldcard') & (data['success'] == 1)])
erfolg_ukcard = len(data[(data['PSP'] == 'UK_Card') & (data['success'] == 1)])
erfolg_simplecard = len(data[(data['PSP'] == 'Simplecard') & (data['success'] == 1)])

# Erfolge pro card
erfolg_master = len(data[(data['card'] == 'Master') & (data['success'] == 1)])
erfolg_diners = len(data[(data['card'] == 'Diners') & (data['success'] == 1)])
erfolg_visa = len(data[(data['card'] == 'Visa') & (data['success'] == 1)])


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

# Balkendiagramm erstellen
#_plt.figure(num='PSP')
#_plt.bar(success_psp.index, success_psp.values, width=0.3, edgecolor='black', linewidth=1.2)
#_plt.xlabel('PSP')
#_plt.ylabel('Anzahl')
#_plt.title("erfolgreiche Transaktionen je PSP")

#_plt.figure(num='card')
#_plt.bar(success_card.index, success_card.values, width=0.3, edgecolor='black', linewidth=1.2)
#_plt.xlabel('Kartenanbieter')
#_plt.ylabel('Anzahl')
#_plt.title("erfolgreiche Transaktionen je Karte")

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
# Filtern nach dem Land
#_germany_transactions = data[data['country'] == 'Germany']['amount']
#_austria_transactions = data[data['country'] == 'Austria']['amount']
#_switzerland_transactions = data[data['country'] == 'Switzerland']['amount']
# Filtern nach dem Erfolg
erfolg_transactions = data[data['success'] == 1]['gebuehr']
miserfolg_transactions = data[data['success'] == 0]['gebuehr']

# Daten plotten
plt.figure(num='gebuehr_erfolg_miserfolg')
plt.hist(erfolg_transactions, bins=20, edgecolor='black', linewidth=1.2, label = 'Erfolg')  # Anzahl der Bins anpassen
plt.hist(miserfolg_transactions, bins=20, edgecolor='black', linewidth=1.2, label = 'Miserfolg')  # Anzahl der Bins anpassen
plt.xlabel('Gebuehr')
plt.ylabel('Anzahl')
plt.legend()



#_plt.figure(num='amount')
#_plt.subplot(2, 2, 1) 
#_plt.hist(data['amount'], bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
#_plt.xlabel('Transaktionsbetrag gesamt')
#_plt.ylabel('Anzahl')
#_plt.subplot(2, 2, 2) 
#_plt.hist(germany_transactions, bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
#_plt.xlabel('Transaktionsbetrag Deutschland')
#_plt.ylabel('Anzahl')
#_plt.subplot(2, 2, 3) 
#_plt.hist(austria_transactions, bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
#_plt.xlabel('Transaktionsbetrag Australien')
#_plt.ylabel('Anzahl')
#_plt.subplot(2, 2, 4) 
#_plt.hist(switzerland_transactions, bins=20, edgecolor='black', linewidth=1.2)  # Anzahl der Bins anpassen
#_plt.xlabel('Transaktionsbetrag Schweiz')
#_plt.ylabel('Anzahl')
# Abstand zwischen den Subplots anpassen
#_plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Streudiagramm erstellen
#_plt.figure(num='amount_gebuehr')
#_plt.scatter(data['amount'], data['gebuehr'], s=1, color='black')
#_plt.xlabel('Transaktionsbetrag')
#_plt.ylabel('Gebuehr')

# Erstellen eines Zeitreihenplots
# Umwandlung der Spalte 'Zeitstempel' in ein DateTime-Format
#_data['tmsp'] = pd.to_datetime(data['tmsp'])
# Gruppieren nach Tag und Zählen der Transaktionen pro Tag
#_daily_transactions = data.groupby(data['tmsp'].dt.date)['amount'].count()
# Zeitreihenplot der Anzahl der Transaktionen pro Tag erstellen
#_fig, ax = plt.subplots(figsize=(10, 6))  # figsize anpassen
#_plt.plot(daily_transactions.index, daily_transactions.values)
#_plt.xlabel('Zeit in Tagen')
#_plt.ylabel('Anzahl Transaktionen')


# Boxplot erstellen
#_plt.figure(num='Transaktionen pro Land')
#_plt.boxplot([data[data['country'] == 'Germany']['amount'],
             #_data[data['country'] == 'Austria']['amount'],
             #_data[data['country'] == 'Switzerland']['amount']])
# Achsentitel hinzufügen
#_plt.xlabel('Land')
#_plt.ylabel('Transaktionsbetrag')
# Länderbeschriftungen hinzufügen
#_plt.xticks([1, 2, 3], ['Deutschland', 'Australien', 'Schweiz'])


# Korrelationsmatrix berechnen
#_correlation_matrix = data.corr()
# Heatmap erstellen
#_sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')