import os
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from pandas import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

donnees = []
avis = []

columns_name = ['commentaire', 'type_de_commentaire']

def pretraitement_texte(texte):
    try:
        tokens = word_tokenize(texte)
        tokens = [mot for mot in tokens if mot not in string.punctuation]
        mots_vides = set(stopwords.words('english'))
        tokens = [mot for mot in tokens if mot.lower() not in mots_vides]
        lemmatiseur = WordNetLemmatizer()
        tokens = [lemmatiseur.lemmatize(mot) for mot in tokens]
        return " ".join(tokens)
    except Exception as e:
        print("Une erreur s'est produite lors du prétraitement du texte :", e)
        return ""

def getData(chemin,type_commentaire):
    try:
        for fichier in os.listdir(chemin):
            if fichier.endswith(".txt"):
                chemin_fichier = os.path.join(chemin, fichier)
                with open(chemin_fichier, "r", encoding="utf-8") as f:
                    lignes = ([ligne.strip() for ligne in f.readlines()])
                    texte_complet = " ".join(lignes)
                    donnees.append(texte_complet)
                    avis.append(type_commentaire)
    except Exception as e:
        print("Une erreur s'est produite lors de la récupération des données :", e)

try:
    getData("txt_sentoken/neg","neg")
    getData("txt_sentoken/pos","pos")
    data = DataFrame({"commentaire": donnees, "nature_du_commentaire": avis})
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    data['commentaire'] = data['commentaire'].apply(pretraitement_texte)

    X = data["commentaire"]
    y = data["nature_du_commentaire"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    vectorizer = CountVectorizer(lowercase=True)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    naive_bayes = GaussianNB()

    naive_bayes.fit(X_train_vectorized.toarray(), y_train)


    y_pred = naive_bayes.predict(X_test_vectorized.toarray())

    f1 = f1_score(y_test, y_pred, pos_label="pos")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,  pos_label="pos")
    confusion_mat = confusion_matrix(y_test,y_pred)

    print("f1_score:", f1)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Confusion Matrix:", confusion_mat)

    # Affichage de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.title('Matrice de Confusion')
    plt.show()

    # Affichage de la distribution des classes
    plt.figure(figsize=(8, 6))
    sns.countplot(x='nature_du_commentaire', data=data)
    plt.xlabel('Nature du commentaire')
    plt.ylabel('Nombre de commentaires')
    plt.title('Distribution des classes')
    plt.show()

    # Affichage de la longueur des commentaires
    data['longueur_commentaire'] = data['commentaire'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x='longueur_commentaire', bins=20, kde=True)
    plt.xlabel('Longueur du commentaire')
    plt.ylabel('Nombre de commentaires')
    plt.title('Distribution de la longueur des commentaires')
    plt.show()

except Exception as e:
    print("Une erreur s'est produite :", e)
