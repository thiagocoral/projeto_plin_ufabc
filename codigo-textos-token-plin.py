from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
import pandas as pd
import json
import os
import numpy as np
from nltk.stem import WordNetLemmatizer


def extract_df(url):

    page = requests.get(url)

    web_scraped = BeautifulSoup(page.text,"html.parser")

    #tech_info = web_scraped.find(id="content-text__container")
    string_text = ""
    string_representante = ""
    for p in web_scraped.find_all("p",class_="content-text__container"):
        string_text += p.text.lower()
        if p.text[1] == "–":
            string_representante += p.text.lower()


    def tokenizar_texto(string_tonekiner):
        tokens=word_tokenize(string_tonekiner) 

        tokens_without_sw = [word for word in tokens if not word.lower() in stopwords.words("portuguese")]

        tokens_palavras=[w for w in tokens_without_sw if w.isalpha()]

        # lemmatizer = WordNetLemmatizer()
        # newwords = [lemmatizer.lemmatize(word) for word in tokens_palavras]

        fdist=nltk.FreqDist(string_tonekiner)


        fdist_sort=fdist.most_common(50)

        ######### LEI DE ZIPF ##########
        num_types=len(set(tokens_palavras))

        fdist=nltk.FreqDist(tokens_palavras)
        fdist_sort=fdist.most_common(num_types)

        # cria uma lista dos types
        types=[w for (w,f) in fdist_sort ]

        # cria uma lista com as frequências
        freqs=[f for (w,f) in fdist_sort ]

        # cria uma lista com as posições no ranking
        ranks=list(range(1,num_types+1))

        # calcula o produto f*r
        produto= np.array(freqs)*np.array(ranks)

        df = pd.DataFrame(list(zip(types, freqs, ranks, produto)),
                        columns =['Type','Frequência(f)','Posto(r)',"f*r"])
        return df

    df_text_full = tokenizar_texto(string_text)
    df_text_representante = tokenizar_texto(string_representante)
    return df_text_full,df_text_representante



def extract_all_urls():
    with open("PLNLinks.txt","r") as url_file:
        lines = url_file.readlines()
        for line in lines:
            line,target = line.split(",")
            df_text_full,df_text_representante = extract_df(line[0:len(line)])
            #print(df_text_full,df_text_representante)

            ### Verifica se o arquivo existe, se não ele é criado
            if not os.path.isfile("dfs_texts.json"):
                start_data = {
                    "texts": []
                }
                with open("dfs_texts.json","w",encoding='utf-8') as file:
                    json.dump(start_data,file)
                print("file dfs_texts.json created")

            ### Função de escrever os DFs no JSON
            def write_to_json():
                with open("dfs_texts.json","r+", encoding='utf-8') as file:
                    full_file = json.load(file)
                    new_data = {"url":line[0:len(line)-1],"df_text_full":df_text_full.to_dict(),"df_text_representante":df_text_representante.to_dict(),"target":target}
                    full_file["texts"].append(new_data)
                    file.seek(0)
                    json.dump(full_file, file, indent = 4, ensure_ascii=False)

            write_to_json()

extract_all_urls()
