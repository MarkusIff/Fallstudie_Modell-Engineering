# -*- coding: utf-8 -*-
"""
erstellen eines maschinellen Lernmodells
"""

# Vorbereitung

# Biblipotheken importieren
import explorative_datenanalyse as data
import pandas as pd

# bereinigte Daten einlesen

df = pd.read_excel(data.ausgabe + '/' + data.ausgabedatei)