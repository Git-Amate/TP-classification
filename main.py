import os
import seaborn as sns
import re
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from pandas import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score , confusion_matrix



# Liste pour stocker les données de chaque fichier
donnees = []
avis = []

#chemin = "txt_sentoken/neg"


def pretraitement_texte(texte):
    tokens = word_tokenize(texte)
    tokens = [mot for mot in tokens if mot not in string.punctuation]
    #print(string.punctuation)
    mots_vides = set(stopwords.words('english'))
    #print(mots_vides)
    tokens = [mot for mot in tokens if mot.lower() not in mots_vides]
    lemmatiseur = WordNetLemmatizer()
    tokens = [lemmatiseur.lemmatize(mot) for mot in tokens]
    #print(tokens)
    return " ".join(tokens)

def getData(chemin,type_commentaire):
    for fichier in os.listdir(chemin):
        if fichier.endswith(".txt"):
            chemin_fichier = os.path.join(chemin, fichier)
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                texte_complet = " ".join([ligne.strip() for ligne in f.readlines()])
                donnees.append(texte_complet)
                avis.append(type_commentaire)


getData("txt_sentoken/neg","neg")
getData("txt_sentoken/pos","pos")
data = DataFrame({"commentaire": donnees, "nature_du_commentaire": avis})

# Afficher le nombe de commentaires négatifs et positifs
print(data['nature_du_commentaire'].value_counts())

# Mélanger le DataFrame
data = data.sample(frac=1)
data.reset_index(drop=True, inplace=True)

#Pré-traitement
data['commentaire'] = data['commentaire'].apply(pretraitement_texte)



# X valeur d'apprentissage
X = data["commentaire"]

# y valeur de prédiction
y = data["nature_du_commentaire"]

# entrainement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#print(X_train.head())


vectorizer = CountVectorizer(lowercase=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
#print(X_train_vectorized)
#print(vectorizer.vocabulary_)
X_test_vectorized = vectorizer.transform(X_test)


#Instancier le modèle SVM
svm_classifier = SVC(kernel='linear')

#Entraîner le modèle sur les données d'entraînement vectorisé
print("apprentissage")
svm_classifier.fit(X_train_vectorized, y_train)
print("fin apprentissage")


#Faire des prédictions sur l'ensemble de test vectorisé
y_pred = svm_classifier.predict(X_test_vectorized)

'''

# Instancier le classificateur bayésien naïf gaussien
naive_bayes_classifier = GaussianNB()

# Entraîner le modèle sur les données d'entraînement normalisées
naive_bayes_classifier.fit(X_train_vectorized.toarray(), y_train)

# Faire des prédictions sur l'ensemble de test normalisé
y_pred = naive_bayes_classifier.predict(X_test_vectorized.toarray())
'''

#Évaluer la performance du modèle
f1 = f1_score(y_test, y_pred, pos_label="pos")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,  pos_label="pos")
confusion_matrix = confusion_matrix(y_test,y_pred)
print("f1_score:", f1)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("confusion_matrix:", confusion_matrix)


