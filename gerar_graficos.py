import json
from nltk.corpus.reader import wordlist
from numpy.core.fromnumeric import shape
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


jsons = ["dfs_texts2019_2020_using11.json","dfs_texts2019_2020_using101.json",
    "dfs_texts2019_using11.json","dfs_texts2019_using101.json",
    "dfs_textsCarille_using11.json","dfs_textsCarille_using101.json",
    "dfs_textsCoelho_using11.json","dfs_textsCoelho_using101.json",
    "dfs_textsMancini_using11.json","dfs_textsMancini_using101.json",
    "dfs_textsTiago_using11.json","dfs_textsTiago_using101.json"]

df_testes = ["df_text_full","df_text_representante"]
for arquivo in jsons:
    for df_a_ser_testado in df_testes:

        caminho = "C:/Users/thico/Machine Learning/PLIN/projeto_plin_ufabc/dfs_pedrao/"+arquivo

        file = open(caminho,encoding='utf-8')
        file_json = json.load(file)

        dict = file_json["texts"]

        #######
        ####### REALIZAR STEMMING
        #######


        #gerando bag of words
        target_list = []
        word_list = []



        for register in dict:
            #df_text_representante
            #df_text_full
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
            array_all_frequencies.append(array_register)

        # df = pd.DataFrame(data=array_all_frequencies,columns=word_list)

        # print(df.head())
        # freqs = df.sum(axis=0)

        # df_freq = pd.DataFrame(freqs,columns=word_list)

        freq = sum(array_all_frequencies)

        to_df = {"word": word_list,"freq":freq}
        df = pd.DataFrame(to_df)

        df.sort_values(by="freq",ascending=False,inplace=True)
        df = df.head(50)

        titulo = df_a_ser_testado+" - "+arquivo.split(".")[0]
        #print("Titulo:",titulo)
        x_pos = np.arange(len(freq))
        f=plt.figure()
        f.set_figwidth(20)
        f.set_figheight(10)

        sns.barplot(x="word",y="freq",data=df,palette="coolwarm")

        #Blues_d
        #rocket

        plt.ylabel('Frequência')
        plt.xlabel('Palavra')
        plt.xticks(rotation=90)
        plt.title(titulo)
        plt.savefig("C:/Users/thico/Machine Learning/PLIN/projeto_plin_ufabc/registros_feature_importances/barplots/"+titulo+(".png"))
        print(titulo,"types:",len(word_list),"tokens:",sum(freq),"dLexical:",len(word_list)/sum(freq))