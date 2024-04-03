import os
import re
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
from matplotlib import colors
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from pandas import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Liste pour stocker les données de chaque fichier
donnees = []
avis = []

columns_name = ['commentaire', 'type_de_commentaire']
#chemin = "txt_sentoken/neg"

def getData(chemin,type_commentaire):
    for fichier in os.listdir(chemin):
        if fichier.endswith(".txt"):
            chemin_fichier = os.path.join(chemin, fichier)
            with open(chemin_fichier, "r", encoding="utf-8") as f:
                lignes = f.readlines()
                for ligne in lignes:
                    donnees.append(ligne.rstrip())  # Supprimer les sauts de ligne à droite
                    avis.append(type_commentaire)

getData("txt_sentoken/neg","neg")
getData("txt_sentoken/pos","pos")


data = DataFrame({"commentaire": donnees, "nature_du_commentaire": avis})
# Mélanger le DataFrame
data = data.sample(frac=1)
data.reset_index(drop=True, inplace=True)
#print(data.head())

# X valeur d'apprentissage
X = data["commentaire"]

# y valeur de prédiction
y = data["nature_du_commentaire"]

# entrainement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



vectorizer = CountVectorizer(lowercase=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
#print(vectorizer.vocabulary_)
X_test_vectorized = vectorizer.transform(X_test)

'''
#Instancier le modèle SVM
svm_classifier = SVC()

#Entraîner le modèle sur les données d'entraînement normalisées
svm_classifier.fit(X_train_vectorized, y_train)


#Faire des prédictions sur l'ensemble de test normalisé
y_pred = svm_classifier.predict(X_test_vectorized)
'''

# Instancier le classificateur bayésien naïf gaussien
naive_bayes_classifier = GaussianNB()

# Entraîner le modèle sur les données d'entraînement normalisées
naive_bayes_classifier.fit(X_train_vectorized.toarray(), y_train)

# Faire des prédictions sur l'ensemble de test normalisé
y_pred = naive_bayes_classifier.predict(X_test_vectorized.toarray())

#Évaluer la performance du modèle
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,  pos_label="pos")
print("Accuracy:", accuracy)
print("Precision:", precision)

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_vectorized.toarray())

disp = DecisionBoundaryDisplay.from_estimator(
    naive_bayes_classifier, X_reduced, response_method="predict", xlabel="message", ylabel=type,alpha=0.5 )

# Vectorisation des textes
disp.ax_.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y , edgecolor="k")
plt.show()

