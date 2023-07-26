# -*- coding: utf-8 -*-
"""
erstellen eines maschinellen Lernmodells
"""

# Vorbereitung

# Biblipotheken importieren
#import explorative_datenanalyse as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sys

''''
Konsolenabfrage mit zulässigen Wörtern
eingabe = Abfrage in Konsole
zulässige Wörter = String mit zulässigen Wörtern für die Abfrage
'''
def eingabe_mit_zulaessigen_woertern(prompt, zulaessige_woerter):
    while True:
        eingabe = input(prompt)
        if eingabe in zulaessige_woerter:
            return eingabe
        else:
            print("Ungültige Eingabe. Bitte versuchen Sie es erneut.")
            

# bereinigte Daten einlesen
data_ohne_duplikate = pd.read_excel('C:\Datensaetze generieren/bereinigter_Datensatz_ohne_Duplikate.xlsx')
data_mit_duplikate = pd.read_excel('C:\Datensaetze generieren/bereinigter_Datensatz.xlsx')    

# zukünftige Daten einlesen
future_data = pd.read_excel('C:\Datensaetze generieren/future_data.xlsx')   

# Definieren Sie die Features (unabhängigen Variablen) und die Zielvariable (Kreditkarte)
X = data_mit_duplikate.drop(columns=['tmsp','success','gebuehr', 'laufende Nr.'])
y = np.ravel(data_mit_duplikate[['success']])

# Umwandlung in Dummy-Variablen (One-Hot-Encoding)
X = pd.get_dummies(X, columns=['country', 'PSP', 'card'])

# Teilen Sie die Daten in Trainings- und Testdaten auf (z.B. 80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Normale logistic Regression'''

# logistic Regression
log_reg_model = LogisticRegression(C=0.1, penalty='l2', max_iter=500, class_weight='balanced', solver='lbfgs')

# Trainieren Sie die logistische Regression
log_reg_model.fit(X_train, y_train)

# Evaluieren Sie die Leistung des Modells auf den Testdaten (optional)
accuracy = log_reg_model.score(X_test, y_test)
print('Genauigkeit:', accuracy)

# Speichern Sie das trainierte Modell in einer Datei
import pickle
with open('trainiertes_logistisches_modell.pkl', 'wb') as file:
    pickle.dump(log_reg_model, file)
    
# Vorhersage für die Testdaten
y_pred = np.ravel(log_reg_model.predict(X_test))    
classification_report_result = classification_report(y_test, y_pred)
print(f'Classification Report success:\n{classification_report_result}')


'''Vorhersage machen'''

# Dataframe aufgrund von Benutzerangaben erstellen
zulaessige_woerter_kreditkarte = ['Visa','Master','Diners']
kreditkarte = eingabe_mit_zulaessigen_woertern("Name der Kreditkarte: ", zulaessige_woerter_kreditkarte)
zulaessige_woerte_land = ['Deutschland','Schweitz','Australien']
land = eingabe_mit_zulaessigen_woertern("Land der Transaktion: ", zulaessige_woerte_land)
zulaessige_woerter_secured = ['ja', 'nein']
secured =  eingabe_mit_zulaessigen_woertern("benutzen Sie die 3D Identifizierung: ", zulaessige_woerter_secured)
betrag = input("geben Sie den Betrag an: ")

future_data['amount'] = int(betrag)

if secured == "ja":
    future_data['3D_secured'] = 1
else:
    future_data['3D_secured'] = 0
    
if land == "Australien":
    future_data['country_Austria'] = 1
    future_data['country_Germany'] = 0
    future_data['country_Switzerland'] = 0
elif land == "Deutschland":
    future_data['country_Austria'] = 0
    future_data['country_Germany'] = 1
    future_data['country_Switzerland'] = 0
else:
    future_data['country_Austria'] = 0
    future_data['country_Germany'] = 0
    future_data['country_Switzerland'] = 1
    
if kreditkarte == "Visa":
    future_data['card_Diners'] = 0
    future_data['card_Master'] = 0
    future_data['card_Visa'] = 1
elif kreditkarte == "Master":
    future_data['card_Diners'] = 0
    future_data['card_Master'] = 1
    future_data['card_Visa'] = 0
else:
    future_data['card_Diners'] = 1
    future_data['card_Master'] = 0
    future_data['card_Visa'] = 0

predictions_success = log_reg_model.predict(future_data)
probabilities = log_reg_model.predict_proba(future_data)

''' Vorhersagen auswerten'''
# Umwandeln der Arrays VOrhersage und Wahrscheinlichkeiten in Dataframes
predictions_success = pd.DataFrame(predictions_success, columns=['success'])
probabilities = pd.DataFrame(probabilities, columns=['Transaktion gescheitert', 'Transaktion erfolgreich'])

# Dteframe zusammenfügen
auswertung = pd.concat([future_data, predictions_success, probabilities], axis=1)

# Servicegebuehren der PSPs in Auswertung in Spalte Gebühr hinzufügen
auswertung['gebuehr'] = 0
auswertung.loc[(auswertung['PSP_Moneycard'] == 1) & (auswertung['success'] == 1), 'gebuehr'] = 5
auswertung.loc[(auswertung['PSP_Moneycard'] == 1) & (auswertung['success'] == 0), 'gebuehr'] = 2
auswertung.loc[(auswertung['PSP_Goldcard'] == 1) & (auswertung['success'] == 1), 'gebuehr'] = 10
auswertung.loc[(auswertung['PSP_Goldcard'] == 1) & (auswertung['success'] == 0), 'gebuehr'] = 5
auswertung.loc[(auswertung['PSP_UK_Card'] == 1) & (auswertung['success'] == 1), 'gebuehr'] = 3
auswertung.loc[(auswertung['PSP_UK_Card'] == 1) & (auswertung['success'] == 0), 'gebuehr'] = 1
auswertung.loc[(auswertung['PSP_Simplecard'] == 1) & (auswertung['success'] == 1), 'gebuehr'] = 1
auswertung.loc[(auswertung['PSP_Simplecard'] == 1) & (auswertung['success'] == 0), 'gebuehr'] = 0.5

# DataFrame in eine Excel-Datei mit Formatierung speichern
with pd.ExcelWriter('C:\Datensaetze generieren\Vorhersage.xlsx', engine='xlsxwriter') as writer:
    auswertung.to_excel(writer, sheet_name='Vorhersage', index=False)

    # Zugriff auf das Excel-Workbook und das Arbeitsblatt
    workbook = writer.book
    worksheet = writer.sheets['Vorhersage']

    # Definiere eine Formatierung für grünen Hintergrund
    format_green = workbook.add_format({'bg_color': 'green'})

    # PSP Zuordnung
    zuordnung = False
    for row_idx, row in auswertung.iterrows():
        # Setze grüner Hintergrund für die Zeile basierend auf dem Wert in der Spalte
        if row['Transaktion erfolgreich'] > 0.7:
            worksheet.set_row(row_idx + 1, None, format_green)
            zuordnung = True
            sys.exit()
        elif row['Transaktion erfolgreich']  > 0.6 and row['gebuehr']  < 10:
            worksheet.set_row(row_idx + 1, None, format_green)
            zuordnung = True
            sys.exit()
        elif row['Transaktion erfolgreich']  > 0.5 and row['gebuehr']  < 2:
            worksheet.set_row(row_idx + 1, None, format_green)
            zuordnung = True
            sys.exit()
    
# Wenn PSP Zuordnung fehlschlägt manuell Zuordnnung
if zuordnung == False:
    print('manuelle PSP Zuordnung erforderlich')


