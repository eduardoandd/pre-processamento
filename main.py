import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

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
# df.loc[df['age'] < 0]
# df2=df.drop('age',axis= 1)
# # ou
# df3=df.drop(df[df['age'] < 0 ].index)
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

X_credit_treinamento,X_credit_teste,y_credit_treinamento,y_credit_teste=train_test_split(X,y,test_size=0.25,random_state=0)

with open('credit.pkl', mode= 'wb') as f:
    pickle.dump([X_credit_treinamento,y_credit_treinamento,X_credit_teste,y_credit_teste],f)

# ==== CENSO ====
df_census= pd.read_csv('census.csv')
df_census.describe()
df_census.isnull().sum()

#Visualização
np.unique(df_census['income'],return_counts=True)
sns.countplot(x=df_census['income'],palette='dark')
plt.hist(x=df_census['age'])
plt.hist(x=df_census['education-num'])
plt.hist(x=df_census['hour-per-week'])
grafico=px.treemap(df_census,path=['occupation','relationship','age'])
grafico=px.parallel_categories(df_census,dimensions=['workclass','occupation','income'])
grafico=px.parallel_categories(df_census,dimensions=['education','income'])

#divisão entre previsores
X_census=df_census.iloc[:,0:14].values
y_census=df_census.iloc[:,-1].values

#tratamento de dados categoricos
#labelencoder
label_encoder=LabelEncoder()
teste= label_encoder.fit_transform(X_census[:,1])

label_encoder_workclass=LabelEncoder()
label_encoder_education=LabelEncoder()
label_encoder_marital=LabelEncoder()
label_encoder_occupation=LabelEncoder()
label_encoder_relationship=LabelEncoder()
label_encoder_race=LabelEncoder()
label_encoder_sex=LabelEncoder()
label_encoder_country=LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_workclass.fit_transform(X_census[:,3])
X_census[:,5] = label_encoder_workclass.fit_transform(X_census[:,5])
X_census[:,6] = label_encoder_workclass.fit_transform(X_census[:,6])
X_census[:,7] = label_encoder_workclass.fit_transform(X_census[:,7])
X_census[:,8] = label_encoder_workclass.fit_transform(X_census[:,8])
X_census[:,9] = label_encoder_workclass.fit_transform(X_census[:,9])
X_census[:,13] = label_encoder_workclass.fit_transform(X_census[:,13])


#OneHotEncoder


# carros = ["Ford Fiesta", "Volkswagen Gol", "Chevrolet Onix", "Hyundai HB20", "Toyota Corolla", "Honda Civic", "Renault Sandero", "Fiat Palio", "Nissan Kicks", "Jeep Renegade"]
# cidades = ["São Paulo", "Rio de Janeiro", "São Paulo", "Porto Alegre", "Curitiba", "Brasília", "Salvador", "Recife", "Fortaleza", "Rio de Janeiro"]
# precos = [40000, 35000, 45000, 50000, 80000, 75000, 30000, 20000, 90000, 95000]
# df_carros = pd.DataFrame({"carro": carros, "cidade": cidades, "preco": precos})

# X_carros=df_carros.iloc[:,0:2].values
# y_carros=df_carros.iloc[:,2].values
# le_carro=LabelEncoder()
# le_cidade=LabelEncoder()
# X_carros[:,0]=le_carro.fit_transform(X_carros[:,0])
# X_carros[:,1]=le_cidade.fit_transform(X_carros[:,1])

# onehotencoder_carros=ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),[0,1])],remainder='passthrough')
# X_carros=onehotencoder_carros.fit_transform(X_carros).toarray()
# scaler_carros=StandardScaler()
# X_carros=scaler_carros.fit_transform(X_carros)
# X_carros_treinamento,X_carros_teste,y_carros_treinamento,y_carros_teste=train_test_split(X_carros,y_carros,test_size=0.15,random_state=0)


onehotencoder_census= ColumnTransformer(transformers=[('OneHot',OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')    
X_census=onehotencoder_census.fit_transform(X_census).toarray()
scaler_census = StandardScaler()
X_census=scaler_census.fit_transform(X_census)
X_census_treinamento,X_census_teste,y_census_treinamento,y_census_teste=train_test_split(X_census,y_census,test_size=0.15,random_state=0)

with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento,y_census_treinamento,X_census_teste,y_census_teste],f)






