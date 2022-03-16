# Programa de aprendizado de MVS
# Seguindo o tutorial https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python?utm_source=adwords_ppc&utm_medium=cpc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=278443377095&utm_targetid=aud-299261629574:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=9074200&gclid=Cj0KCQiArt6PBhCoARIsAMF5wahFqVnOosUMNo9LKCjaG-cqliBbQ9AID7PFET0P1-pYzzyjcoHNPD4aArngEALw_wcB
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics


os.system('cls' if os.name == 'nt' else 'clear')    # Limpa terminal
# Importando o dataset
cancer = datasets.load_breast_cancer()  # dataset padrão sobre cancer
                                        # 569 'samples' com 30 'features'
# Separando os dados em treinamento e validaçao
# 70% para treinamento, 30% para validação
X_treino, X_valid, y_treino, y_valid = train_test_split(cancer.data,cancer.target,test_size=0.3,train_size=0.7)
# Criando a MVS
for Kernel in ('linear','poly','rbf'):
    Classificador=SVC(kernel=Kernel)  # Criando SVM com kernel linear
                                            # K(x,xi) = sum(x*xi)
    # Treinamento da MVS
    Classificador.fit(X_treino,y_treino)
    # Validação da MVS
    y_predicao=Classificador.predict(X_valid)
    #Avaliação da precisão
    print("Precisão: ",metrics.precision_score(y_predicao,y_valid))





