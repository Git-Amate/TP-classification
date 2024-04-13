import os
import re
from keras import layers
from keras import losses
from numpy import *
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.svm import SVC
from pandas import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score , confusion_matrix

# Liste pour stocker les données de chaque fichier
donnees = []
avis = []

columns_name = ['commentaire', 'type_de_commentaire']
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
                lignes = ([ligne.strip() for ligne in f.readlines()])
                texte_complet = " ".join(lignes)
                donnees.append(texte_complet)
                avis.append(type_commentaire)

getData("txt_sentoken/neg","neg")
getData("txt_sentoken/pos","pos")

data = DataFrame({"commentaire": donnees, "nature_du_commentaire": avis})
print(data['nature_du_commentaire'].value_counts())
# Mélanger le DataFrame
data = data.sample(frac=1)
data.reset_index(drop=True, inplace=True)
data['commentaire'] = data['commentaire'].apply(pretraitement_texte)
data['nature_du_commentaire'] = data['nature_du_commentaire'].replace({'pos': 1, 'neg': 0})


# X valeur d'apprentissage
X = data["commentaire"]

# y valeur de prédiction
y = data["nature_du_commentaire"].values

# entrainement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Paramètres de tokenisation
vocab_size = 10000
embedding_dim = 50
max_length = 150
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

#Tokenisation des phrases
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
padded_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=max_length, padding=padding_type, truncating=trunc_type)
padded_test = tf.keras.preprocessing.sequence.pad_sequences(sequences_test, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Création du modèle LSTM bidirectionnel
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
model.fit(padded_train, y_train, epochs=10, validation_data=(padded_test, y_test))

# Evaluation du modèle
loss, accuracy = model.evaluate(padded_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

