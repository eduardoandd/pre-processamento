import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle


df_census= pd.read_csv('census.csv')
df_census.describe()
df_census.isnull().sum()

# ============= Visualização =============
np.unique(df_census['income'],return_counts=True)
sns.countplot(x=df_census['income'],palette='dark')
plt.hist(x=df_census['age'])
plt.hist(x=df_census['education-num'])
plt.hist(x=df_census['hour-per-week'])
grafico=px.treemap(df_census,path=['occupation','relationship','age'])
grafico=px.parallel_categories(df_census,dimensions=['workclass','occupation','income'])
grafico=px.parallel_categories(df_census,dimensions=['education','income'])

# ============= divisão entre previsores e classe =============
X=df_census.iloc[:,0:-1].values
y=df_census.iloc[:,-1].values

# ============= tratamento de atributos categóricos =============

#labelEncoder
label_encoder_teste = LabelEncoder()
teste=label_encoder_teste.fit_transform(X[:,1])
indices=[]

for i in range(X.shape[1]):
    if X[:,i].dtype == 'object':
        X[:,i] = label_encoder_teste.fit_transform(X[:,i])
    
    if df_census.dtypes[i] == 'object':
        indices.append(i)

#OneHotEncoder
# carros = ['Uno','Palio','Celta','Celta','Palio','Ford Ka','A20','Ford Ka']
# # # precos=[15000,2000,5000,4000,3000]
# df_carros = pd.DataFrame({"carro": carros})
# X_carro = df_carros.iloc[:,0:].values
# label_encoder_teste.fit_transform(X_carro[:])

# one_hot_encoder=ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),[0])],remainder='passthrough')
# one_hot_encoder.fit_transform(X_carro).toarray()

one_hot_encoder=ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),indices)],remainder='passthrough')
X=one_hot_encoder.fit_transform(X).toarray()

# ============= Escalonamento de valores =============

scaler_census=StandardScaler()
X=scaler_census.fit_transform(X)

# ============= Divisão entre base de treinamento e teste =============
X_treinamento,X_teste,y_treinamento,y_teste= train_test_split(X,y,test_size=0.15,random_state=0)

# ============= Salvando variáveis em disco =============
with open('census.pkl', mode='wb') as f:
    pickle.dump([X_treinamento,y_treinamento,X_teste,y_teste],f)


