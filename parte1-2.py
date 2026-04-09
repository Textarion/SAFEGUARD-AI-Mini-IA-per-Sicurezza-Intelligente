# SafeGuard AI è un sistema di videosorveglianza alimentato ad intelligenza artificiale, scritto con python e con l'aiuto di diverse
# librerie aiuta ad identificare le masse con potenziali vandalismi e intrusioni, garantendo la sicurezza 
# di un posto archeologico come la valle dei templi.

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
