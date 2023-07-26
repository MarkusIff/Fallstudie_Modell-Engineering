"""
explorative Datenanalyse
"""
# Vorbereitung

# Biblipotheken importieren
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
#import sys

# Onlineshop Daten einlesen
eingabe = input("geben Sie den Speicherort des Datensatzes an: ")
eingabedatei = 'PSP_Jan_Feb_2019.xlsx'
data = pd.read_excel(eingabe + '/' + eingabedatei)

"""
Merkmale generieren und im Dataframe hinzufügen
"""
# zwei Überweisungen in derselben Minute, aus demselben Land und mit demselben Überweisungsbetrag in Spalte Duplikat kennzeichnen
# Konvertieren des Zeitstempels in den DateTime-Datentyp
data['tmsp'] = pd.to_datetime(data['tmsp'])

# Überprüfen auf Duplikate basierend auf den Bedingungen
for index, row in data.iterrows():
    if index < len(data) - 1:
        if data.loc[index,'country'] == data.loc[index + 1,'country'] and data.loc[index,'amount'] == data.loc[index + 1,'amount'] and data.loc[index,'success'] == 0 and data.loc[index,'tmsp'].minute == data.loc[index + 1,'tmsp'].minute:
            if data.loc[index + 1,'success'] == 0:
                data.loc[index,'Duplikat'] = 1
                data.loc[index + 1,'Duplikat'] = 1
            else:
                data.loc[index,'Duplikat'] = 1
# Setzen der fehlenden Werte auf 0
data['Duplikat'] = data['Duplikat'].fillna(0)

# Aufsplitten der Spalte tmsp in Jahr, Montag, Tag, Uhrzeit und Datum
data['tmsp'] = data['tmsp'].astype(str)
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
confidence_interval = 1.95 * (std_counts / np.sqrt(len(hourly_counts)))

# Berechnung des Konfidenzintervalls für die Anzahl der Transaktionen pro Tag
mean_counts_d = day_counts.mean()
std_counts_d = day_counts.std()
confidence_interval_t = 1.95 * (std_counts_d / np.sqrt(len(day_counts)))


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
plt.figure(num='success je card und PSP', figsize=(8, 6))
plt.subplot(2, 1, 1) 
bars_1 = plt.bar(success_psp.index, success_psp.values, width=0.3, edgecolor='black', linewidth=1.2)
# Füge den Text für jeden Balken hinzu
for bar in bars_1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height-100, str(height), ha='center', va='top')    
plt.xlabel('PSP')
plt.ylabel('Anzahl')
plt.subplot(2, 1, 2) 
bars_2 = plt.bar(success_card.index, success_card.values, width=0.3, edgecolor='black', linewidth=1.2)
# Füge den Text für jeden Balken hinzu
for bar in bars_2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height-100, str(height), ha='center', va='top')   
plt.xlabel('Kartenanbieter')
plt.ylabel('Anzahl')


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
plt.ylabel('Anzahl Transaktionen')
# Legende anzeigen
plt.legend()
plt.subplot(3,1,2)
plt.plot(day_counts.index, day_counts.values, marker='o', linestyle='-', label='Transaktionen pro Tag')
plt.fill_between(day_counts.index, mean_counts_d-confidence_interval_t, mean_counts_d+confidence_interval_t, color='gray', alpha=0.3, label='Konfidenzintervall')
# Achsentitel hinzufügen
plt.xlabel('Tag')
plt.ylabel('Anzahl Transaktionen')
# Legende anzeigen
plt.legend()
plt.subplot(3,1,3)
plt.plot(month_counts.index, month_counts.values, marker='o', linestyle='-', label='Transaktionen pro Monat')
# Achsentitel hinzufügen
plt.xlabel('Monat')
plt.ylabel('Anzahl Transaktionen')
# Legende anzeigen
plt.legend()
plt.subplots_adjust(hspace=0.4, wspace=0.1)

# Boxplot erstellen
plt.figure(num='Transaktionen pro Land', figsize=(8, 6))
plt.subplot(3, 1, 1) 
bp = plt.boxplot([data[data['country'] == 'Germany']['amount'],
             data[data['country'] == 'Austria']['amount'],
             data[data['country'] == 'Switzerland']['amount']])
# Werte für den Median, den minimalen und den maximalen Wert
median = bp['medians'][0].get_ydata()[0]
minimum = bp['caps'][0].get_ydata()[0]
maximum = bp['caps'][1].get_ydata()[0]
# Füge den Text an den gewünschten Positionen hinzu
plt.text(0.6, median, str(median), ha='left', va='bottom')
plt.text(0.6, minimum, str(minimum), ha='left', va='bottom')
plt.text(0.6, maximum, str(maximum), ha='left', va='bottom')
# Achsentitel hinzufügen
plt.ylabel('Transaktionsbetrag')
# Länderbeschriftungen hinzufügen
plt.xticks([1, 2, 3], ['Deutschland', 'Australien', 'Schweiz'])
plt.subplot(3, 1, 2) 
plt.boxplot([data[data['card'] == 'Diners']['amount'],
             data[data['card'] == 'Master']['amount'],
             data[data['card'] == 'Visa']['amount']])
# Werte für den Median, den minimalen und den maximalen Wert
median = bp['medians'][0].get_ydata()[0]
minimum = bp['caps'][0].get_ydata()[0]
maximum = bp['caps'][1].get_ydata()[0]
# Füge den Text an den gewünschten Positionen hinzu
plt.text(0.6, median, str(median), ha='left', va='bottom')
plt.text(0.6, minimum, str(minimum), ha='left', va='bottom')
plt.text(0.6, maximum, str(maximum), ha='left', va='bottom')
# Achsentitel hinzufügen
plt.ylabel('Transaktionsbetrag')
# Länderbeschriftungen hinzufügen
plt.xticks([1, 2, 3], ['Diners', 'Master', 'Visa'])
plt.subplot(3, 1, 3) 
plt.boxplot([data[data['PSP'] == 'Goldcard']['amount'],
             data[data['PSP'] == 'Moneycard']['amount'],
             data[data['PSP'] == 'Simplecard']['amount'],
             data[data['PSP'] == 'UK_Card']['amount']])
# Werte für den Median, den minimalen und den maximalen Wert
median = bp['medians'][0].get_ydata()[0]
minimum = bp['caps'][0].get_ydata()[0]
maximum = bp['caps'][1].get_ydata()[0]
# Füge den Text an den gewünschten Positionen hinzu
plt.text(0.5, median, str(median), ha='left', va='bottom')
plt.text(0.6, minimum, str(minimum), ha='left', va='bottom')
plt.text(0.6, maximum, str(maximum), ha='left', va='bottom')
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


"""
Datenaufbereitung
"""

# Formatierung und Umbenennung der Spalten
data.rename(columns={'Spalte1': 'laufende Nr.'}, inplace=True)
data['tmsp'] = pd.to_datetime(data['tmsp'])
data['Duplikat'] = data['Duplikat'].astype(int)

# Auswerten der Datenhistorie
min_zeitstempel = data['tmsp'].min()
max_zeitstempel = data['tmsp'].max()
has_every_minute = (max_zeitstempel - min_zeitstempel) >= pd.Timedelta(minutes=1)
if (max_zeitstempel - min_zeitstempel) >= pd.Timedelta(minutes=1):
    print("Die Datenhistorie ist vollständig.")
else:
    print("Die Datenhistorie ist unvollständig.")
    
# Überprüfen des Dataframe auf leere Einträge
hat_leere_zellen = data.isnull().any().any()
if hat_leere_zellen:
    print("Der Dataframe enthält leere Zellen.")
else:
    print("Der Dataframe enthält keine leeren Zellen.")
    
# Sinnhaftigkeit der Daten überprüfen
# Überprüfen, ob die Spalte "success" nur die Werte 0 und 1 enthält
unique_values = data['success'].unique()
if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
    print("Die Spalte 'success' enthält ausschließlich die Werte 0 und 1.")
else:
    print("Die Spalte 'success' enthält andere Werte als 0 und 1.")  
# Überprüfen, ob die Spalte "PSP" nur die spezifischen Werte enthält
allowed_values = ['UK_Card', 'Moneycard', 'Simplecard', 'Goldcard']
result = data['PSP'].isin(allowed_values).all()
if result:
    print("Die Spalte 'PSP' enthält ausschließlich die erlaubten Werte.")
else:
    print("Die Spalte 'PSP' enthält andere Werte als die erlaubten.")
# Überprüfen, ob die Spalte "PSP" nur die spezifischen Werte enthält
allowed_values = ['Austria', 'Germany', 'Switzerland']
result = data['country'].isin(allowed_values).all()
if result:
    print("Die Spalte 'country' enthält ausschließlich die erlaubten Werte.")
else:
    print("Die Spalte 'country' enthält andere Werte als die erlaubten.")
# Überprüfen, ob die Spalte "PSP" nur die spezifischen Werte enthält
allowed_values = ['Diners', 'Master', 'Visa']
result = data['card'].isin(allowed_values).all()
if result:
    print("Die Spalte 'card' enthält ausschließlich die erlaubten Werte.")
else:
    print("Die Spalte 'card' enthält andere Werte als die erlaubten.")

# Formatierung des Zeitstemples zurück in einen String
data['tmsp'] = data['tmsp'].astype(str)
print(data.dtypes)

# Dataframe generieren bei dem Datensätze mit zwei Überweisungen in derselben Minute, aus demselben Land und mit demselben Überweisungsbetrag entfernt werden
# Daten in Spalte Duplikat bereits markiert
data_2 = data.drop(data[data['Duplikat'] == 1].index)

# nicht notwendige Merkmale entfernen aus data und data_2 entfernen
data = data.drop(['Duplikat','Jahr', 'Monat', 'Tag', 'uhrzeit', 'Hour'], axis=1)
data_2 = data_2.drop(['Duplikat','Jahr', 'Monat', 'Tag', 'uhrzeit', 'Hour'], axis=1)

# Erstellen der Trainings- und Testdatensätze 80 / 20
#X_train, X_test, y_train, y_test = train_test_split(data[['tmsp', 'country', 'amount', 'PSP', '3D_secured', 'card', 'gebuehr']], data['success'], test_size=0.2, random_state=42)
#X_train_ohne_duplikate, X_test_ohne_duplikate, y_train_ohne_duplikate, y_test_ohne_duplikate = train_test_split(data_2[['tmsp', 'country', 'amount', 'PSP', '3D_secured', 'card', 'gebuehr']], data_2['success'], test_size=0.2, random_state=42)


# DataFrame als Excel-Datei für die weitere Bearbeitung speichern
ausgabe = input("geben Sie den Speicherort für die Ausgabe an: ")

data.to_excel(ausgabe + '/bereinigter_Datensatz.xlsx' , index=False)
data_2.to_excel(ausgabe + '/bereinigter_Datensatz_ohne_Duplikate.xlsx', index=False)
#X_train.to_excel(ausgabe + '/X_train.xlsx', index=False)
#X_test.to_excel(ausgabe + '/X_test.xlsx', index=False) 
#y_train.to_excel(ausgabe + '/y_train.xlsx', index=False) 
#y_test.to_excel(ausgabe + '/y_test.xlsx', index=False) 
#X_train_ohne_duplikate.to_excel(ausgabe + '/X_train_ohne_duplikate.xlsx', index=False)
#X_test_ohne_duplikate.to_excel(ausgabe + '/X_test_ohne_duplikate.xlsx', index=False)
#y_train_ohne_duplikate.to_excel(ausgabe + '/y_train_ohne_duplikate.xlsx', index=False) 
#y_test_ohne_duplikate.to_excel(ausgabe + '/y_test_ohne_duplikate.xlsx', index=False) 