import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('credit_data.csv')


# visualização
np.unique(df['default'], return_counts=True) #contagem de cada valor
sns.countplot(x=df['default'])
plt.hist(x=df['age'])
plt.hist(x=df['income'])
plt.hist(x=df['loan'])

grafico=px.scatter_matrix(df,dimensions=['age','income','loan'],color='default')

#tratamento de valores incosistentes
# etapa 1  - fazer a localização de valores incosistentes
df.loc[df['age'] < 0]
df2=df.drop('age',axis= 1)
# ou
df3=df.drop(df[df['age'] < 0 ].index)
# ou
df.loc[df['age'] < 0 ,'age'] =40.92 #mais indicado

#tratamento de valores faltantes
df.isnull().sum
df['age'].fillna(df['age'].mean(),inplace=True)

#divisão entre previsores e classe
X=df.iloc[:,1:4].values
y=df.iloc[:,4].values

#escalonamento dos valores
X[:,0].min(), X[:,1].min(),X[:,2].min()
X[:,0].max(), X[:,1].max(),X[:,2].max()
scaler = StandardScaler()
X= scaler.fit_transform(X)








