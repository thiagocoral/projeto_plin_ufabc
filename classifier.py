import json
from nltk.corpus.reader import wordlist
import pandas as pd
import numpy as np
import random as rand
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import confusion_matrix



caminho = "C:/Users/thico/Machine Learning/PLIN/projeto_plin_ufabc/dfs_pedrao/dfs_texts2019_2020_using11.json"
df_a_ser_testado = "df_text_representante"


#df_text_representante
#df_text_full

file = open(caminho,encoding='utf-8')
file_json = json.load(file)

dict = file_json["texts"]

#######
####### REALIZAR STEMMING
####### https://stackoverflow.com/questions/24647400/what-is-the-best-stemming-method-in-python


#gerando bag of words
target_list = []
word_list = []


for register in dict:
    
    df = pd.DataFrame.from_dict(register[df_a_ser_testado])
    for word in df.Type:
        if word not in word_list:
            word_list.append(word)
    target_list.append(int(register["target"][:-1]))



#gerando arrays com as frequencias
array_all_frequencies = []
for register in dict:
    array_register = np.zeros(len(word_list))
    df = pd.DataFrame.from_dict(register[df_a_ser_testado])
    for i in range(len(df)):
        array_register[word_list.index(df.iloc[i]["Type"])] = df.iloc[i]["Frequência(f)"]
    array_register = tuple(array_register)
    array_all_frequencies.append(array_register)


#shuffle default = True
x_treino, x_teste,y_treino,y_teste = train_test_split(array_all_frequencies,target_list,test_size=0.4)

############################################################ RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier(n_estimators=15, max_depth=3, random_state=1)
classifier.fit(x_treino, y_treino)

print(confusion_matrix(y_treino, classifier.predict(x_treino)))

# Predict Class
y_pred = classifier.predict(x_teste)

# Accuracy 
accuracy = accuracy_score(y_teste, y_pred)
print("Accuracy RandomForestClassifier:",accuracy)
print("f1_score RandomForestClassifier",f1_score(y_teste,y_pred))

print(confusion_matrix(y_teste, y_pred))


print("####################################")

importancia_features = np.array(classifier.feature_importances_)
nome_features = word_list
dados = {"nome_features":nome_features,"importancia_features":importancia_features}
dados_df = pd.DataFrame(dados)
dados_df.sort_values(by=["importancia_features"],ascending=False,inplace=True)
dados_df = dados_df.head(50)


plt.figure(figsize=(6,8))
sns.color_palette("husl", 9)
sns.barplot(x=dados_df['importancia_features'], y=dados_df['nome_features'])
plt.show()

############################################################ GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
#subsample baixo = maior distribuição de feature_importance
classifier = GradientBoostingClassifier(n_estimators=75, learning_rate=0.1,max_depth=15, random_state=0,subsample=0.5)
classifier.fit(x_treino, y_treino)

# Predict Class
y_pred = classifier.predict(x_teste)

# Accuracy 
accuracy = accuracy_score(y_teste, y_pred)
print("Accuracy GradientBoostingClassifier:",accuracy)
print("f1_score GradientBoostingClassifier",f1_score(y_teste,y_pred))
print("####################################")
importancia_features = np.array(classifier.feature_importances_)
nome_features = word_list
dados = {"nome_features":nome_features,"importancia_features":importancia_features}
dados_df = pd.DataFrame(dados)
dados_df.sort_values(by=["importancia_features"],ascending=False,inplace=True)
dados_df = dados_df.head(50)


plt.figure(figsize=(8,6))
sns.color_palette("husl", 9)
sns.barplot(x=dados_df['importancia_features'], y=dados_df['nome_features'])
plt.show()


################################################### MLP
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(500,250,150,50))
classifier.fit(x_treino, y_treino)

# Predict Class
y_pred = classifier.predict(x_teste)

# Accuracy 
accuracy = accuracy_score(y_teste, y_pred)
print("Accuracy MultiLayerPerceptron:",accuracy)
