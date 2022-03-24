import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
os.system('clear')
data=pd.read_csv(r'MVS/breast-cancer.csv')
cancer=pd.DataFrame(data)
y=cancer.diagnosis
list=['id','diagnosis']
X=data.drop(list,axis=1)

encod=LabelEncoder()
y=encod.fit_transform(y)
X_treino, X_valid, y_treino, y_valid = train_test_split(X,y,test_size=0.4,train_size=0.6,stratify=y)
# Criando a MVS
for Kernel in ('linear','poly','rbf'):
    Classificador=SVC(kernel=Kernel)  # Criando SVM com kernel linear
                                            # K(x,xi) = sum(x*xi)
    # Treinamento da MVS
    Classificador.fit(X_treino,y_treino)
    # Validação da MVS
    y_predicao=Classificador.predict(X_valid)
    #Avaliação da precisão
    print("Precisão pelo kernel '",Kernel,"': ",metrics.precision_score(y_predicao,y_valid))

f,ax=plt.subplots(figsize=(18,18))
matrix=np.triu(X.corr())
sns.heatmap(X.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax,mask=matrix)
plt.show()