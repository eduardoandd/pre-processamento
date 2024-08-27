import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler # calculo de padronização
from sklearn.model_selection import train_test_split
import pickle

#========= Exploração dos dados =========
df=pd.read_csv('credit_data.csv')

# ========= Visualização dos dados =========
np.unique(df['default'], return_counts=True)
sns.countplot(x=df['default'],)
plt.hist(x=df['age'])
plt.hist(x=df['income'])
plt.hist(x=df['loan'])
grafico=px.scatter_matrix(df,dimensions=['age','income','loan'],color='default')

#========= tratamento de valores incosistentes =========
df[df['age'] <0]

# Podemos apagar a coluna inteira
df2=df.drop('age',axis=1) # axis = 1 = coluna

# Podemos apagar somente os registos com valores incosistentes
df3= df.drop(df[df['age'] < 0].index)
df3[df3['age'] < 0]

# Podemos preencher os valores manualmente
df['age'].mean()
df[df['age'] > 0]['age'].mean()
df[df['age'] < 0]= df[df['age'] > 0]['age'].mean()
df['age'].mean()

#========= tratamento de valores nulos =========
df.isna().sum()
df.loc[pd.isna(df['age'])]
df['age'].fillna(df['age'].mean(),inplace=True)

#========= Convertendo para inteiro =========
df['default']=df['default'].astype(int)
df['age']=df['age'].astype(int)

#========= divisao entre previsores e classes =========
X= df.iloc[:,1:4].values
y= df.iloc[:,-1].values

#========= escalonamento de valores =========
scaler = StandardScaler()
X=scaler.fit_transform(X)

# ============= Divisão entre base de treinamento e teste =============
X_treinamento,X_teste,y_treinamento,y_teste=train_test_split(X,y,test_size=0.3, random_state=0)

# ============= Salvando variáveis em disco =============
with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_treinamento,y_treinamento,X_teste,y_teste],f)