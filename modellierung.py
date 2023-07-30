"""
erstellen eines maschinellen Lernmodells
Vorbereitung
"""

# Biblipotheken importieren
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import pickle
#from sklearn.model_selection import GridSearchCV

''''
Konsolenabfrage mit zulässigen Wörtern
promt: Eingabeaufforderung
zulässige Wörter: String mit zulässigen Wörtern für die Eingabe
return: Eingabe der Konsolenabfrage
'''
def eingabe_mit_zulaessigen_woertern(prompt, zulaessige_woerter):
    while True:
        eingabe = input(prompt)
        if eingabe in zulaessige_woerter:
            return eingabe
        else:
            print("Ungültige Eingabe. Bitte versuchen Sie es erneut.")
            

# bereinigte Daten einlesen
eingabe = input("Speicherort der Eingabedaten: ")
print('Verfügbare Datensätze: bereinigter_Datensatz_ohne_Duplikate.xlsx oder bereinigter_Datensatz.xlsx')
zulaessige_woerter = ['bereinigter_Datensatz_ohne_Duplikate.xlsx','bereinigter_Datensatz.xlsx']
eingabedatei = eingabe_mit_zulaessigen_woertern("Welcher Datensatz soll geladen werden? ", zulaessige_woerter)

if eingabedatei == 'bereinigter_Datensatz_ohne_Duplikate.xlsx':
    bereinigte_data = pd.read_excel(eingabe + '/bereinigter_Datensatz_ohne_Duplikate.xlsx')
else:
    bereinigte_data = pd.read_excel(eingabe + '/bereinigter_Datensatz.xlsx')    

# Definieren die Features (unabhängigen Variablen) und die Zielvariable
X = bereinigte_data.drop(columns=['tmsp','success','gebuehr', 'laufende Nr.'])
y = np.ravel(bereinigte_data[['success']])

# Umwandlung in Dummy-Variablen (One-Hot-Encoding)
X = pd.get_dummies(X, columns=['country', 'PSP', 'card'])

# Teilen die Daten in Trainings- und Testdaten auf (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Normale logistic Regression
'''
# Abfrage ob Baseline-Modell verwendet werden soll
zulaessige_woerter = ['ja', 'nein']
baseline =  eingabe_mit_zulaessigen_woertern("Soll das Baseline-Modell verwendet werden? ", zulaessige_woerter)

if baseline == "ja":
    # logistic Rregression Baseline-Modell
    log_reg_model = LogisticRegression()
else:
    # logistic Regression nach Hyperparameter Optimierung
    log_reg_model = LogisticRegression(C= 0.1, class_weight= 'none', max_iter= 500, penalty= 'l2', solver= 'newton-cg')

# grid_search führt zu langen Ladezeiten. Ergebnis von grid search in log_reg_model befüllt und damit weitergearbeitet
# Liste von möglichen Parameter
#param_grid = {'C': [0.1, 0.5, 1.0, 5.0, 10.0],
#              'penalty':['l1','l2'],
#              'solver':['liblinear','lbfgs','newton-cg','sag','saga'],
#              'max_iter':[500,1000,10000],
#             'class_weight':['balanced','none']}

# Grid Search Objekt mit Cross-Validation (z.B. 5-Fold Cross-Validation)
#grid_search = GridSearchCV(log_reg_model, param_grid, cv=5)

# Führe die Grid Search durch
#grid_search.fit(X_train, y_train)

# Zeige die besten Hyperparameter-Kombinationen und die entsprechende Leistungsmetrik (Genauigkeit) an
#print("Beste Hyperparameter-Kombination:", grid_search.best_params_)
#print("Beste Leistung (z.B. Genauigkeit):", grid_search.best_score_)

# Modell trainieren
log_reg_model.fit(X_train, y_train)

# Speichern das trainierte Modell in eine Datei
with open('trainiertes_logistisches_modell.pkl', 'wb') as file:
    pickle.dump(log_reg_model, file)
    
# Evaluation des Modells mit den Testdaten
y_pred = np.ravel(log_reg_model.predict(X_test))
auc_score = roc_auc_score(y_test, y_pred)
print("AUC-Score:", auc_score)    
classification_report_result = classification_report(y_test, y_pred)
print(f'Classification Report success:\n{classification_report_result}')


'''
Vorhersage machen
'''
# Future Dataframe erzeugen
columns = ["amount", "3D_secured", "country_Austria", "country_Germany",  "country_Switzerland", "PSP_Goldcard","PSP_Moneycard", "PSP_Simplecard", "PSP_UK_Card", "card_Diners", "card_Master", "card_Visa"]
future_data = pd.DataFrame({col: [] for col in columns})

# Dataframe Future Data befüllen u.a. aufgrund von Benutzerangaben durch Abfrage
print('Um den besten PSP zu wählen, gebe bitte die folgenden Informationen für deine Transaktion ein: ')
ausgabe = input("Speicherort der PSP Vorhersage: ")
zulaessige_woerter = ['Visa','Master','Diners']
kreditkarte = eingabe_mit_zulaessigen_woertern("Name der Kreditkarte: ", zulaessige_woerter)
zulaessige_woerte= ['Deutschland','Schweitz','Australien']
land = eingabe_mit_zulaessigen_woertern("Land der Transaktion: ", zulaessige_woerte)
zulaessige_woerter = ['ja', 'nein']
secured =  eingabe_mit_zulaessigen_woertern("benutzen Sie die 3D Identifizierung: ", zulaessige_woerter)
betrag = input("geben Sie den Betrag an: ")

future_data['amount']= [int(betrag), int(betrag), int(betrag), int(betrag)]

if secured == "ja":
    
    future_data['3D_secured'] = 1
else:
    future_data['3D_secured'] = 0

future_data.loc[0, 'PSP_Goldcard'] = 1 
future_data.loc[1, 'PSP_Goldcard'] = 0 
future_data.loc[2, 'PSP_Goldcard'] = 0 
future_data.loc[3, 'PSP_Goldcard'] = 0

future_data.loc[0, 'PSP_Moneycard'] = 0 
future_data.loc[1, 'PSP_Moneycard'] = 1 
future_data.loc[2, 'PSP_Moneycard'] = 0 
future_data.loc[3, 'PSP_Moneycard'] = 0

future_data.loc[0, 'PSP_Simplecard'] = 0 
future_data.loc[1, 'PSP_Simplecard'] = 0 
future_data.loc[2, 'PSP_Simplecard'] = 1 
future_data.loc[3, 'PSP_Simplecard'] = 0

future_data.loc[0, 'PSP_UK_Card'] = 0 
future_data.loc[1, 'PSP_UK_Card'] = 0 
future_data.loc[2, 'PSP_UK_Card'] = 0 
future_data.loc[3, 'PSP_UK_Card'] = 1  
    
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

'''
Vorhersagen auswerten
'''
# Umwandeln der Arrays Vorhersage und Wahrscheinlichkeiten in Dataframes
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
with pd.ExcelWriter(ausgabe + '\Vorhersage.xlsx', engine='xlsxwriter') as writer:
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
            break
        elif row['Transaktion erfolgreich']  > 0.6 and row['gebuehr']  < 10:
            worksheet.set_row(row_idx + 1, None, format_green)
            zuordnung = True
            break
        elif row['Transaktion erfolgreich']  > 0.5 and row['gebuehr']  < 2:
            worksheet.set_row(row_idx + 1, None, format_green)
            zuordnung = True
            break
        elif row['Transaktion erfolgreich']  < 0.5 and row['gebuehr']  < 1:
            worksheet.set_row(row_idx + 1, None, format_green)
            zuordnung = True
            break
    
# Wenn PSP Zuordnung fehlschlägt manuell Zuordnnung sonst bevorzugten PSP in Konsole ausgeben 
if zuordnung == False:
    print('manuelle PSP Zuordnung erforderlich')
else:
    zu_ueberpruefende_spalten = ['PSP_Goldcard', 'PSP_Moneycard', 'PSP_Simplecard', 'PSP_UK_Card'] # Liste mit den Namen der zu überprüfenden Spalten
    for column in zu_ueberpruefende_spalten:
        if auswertung.loc[row_idx, column] == 1:            
            print('bevorzugter PSP: ' + column)
    


