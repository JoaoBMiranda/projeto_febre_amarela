import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------------------------------------
# Carregar os arquivos .csv em um DataFrame do pandas
#-----------------------------------------------------------

df = pd.read_csv('fa_casoshumanos_1994-2023.csv', sep =';', encoding='latin-1')
df_2 = pd.read_csv('VacinadosFebreAnoxUF.csv', sep=';', encoding='latin-1')
#df.info()

#-----------------------------------------------------------
# ALterando o nome das colunas do df(caso de febre amarela em humanos)
#-----------------------------------------------------------

df.columns = ['id', 'macrorregiao', 'cod_uf', 'estados',
               'cod_mun', 'municipio', 'genero', 'idade',
              'data_inicio_sintomas', 'se_s', 'me_is',
              'dt_ano', 'monitoramento', 'obito', 'dt_obito']

#-----------------------------------------------------------
# Média das idades
#-----------------------------------------------------------

df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
print('--------------------------------------------------')
media_idade = df['idade'].mean().round()
print('Média das idades: ', media_idade)

#-----------------------------------------------------------
# Contagem de casos por macrorregião
#-----------------------------------------------------------

print('--------------------------------------------------')
contagem_macrorregioes = df['macrorregiao'].value_counts()

print('Contagem de Casos por macrorregião:', contagem_macrorregioes)

plt.title('Gráfico de Casos por Macrorregião')
df['macrorregiao'].value_counts().plot(kind='barh', color='skyblue')
plt.xlabel('Quantidade de Casos')
plt.ylabel('Macroregiões')
plt.show()

print('--------------------------------------------------')
#-----------------------------------------------------------
# Contagem de casos por estados
#-----------------------------------------------------------
contagem_estados = df['estados'].value_counts()

print(contagem_estados)
plt.title('Gráfico de Casos por Estado')
df['estados'].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel('Estados')
plt.ylabel('Quantidade de Casos')
plt.show()
print('--------------------------------------------------')
#-----------------------------------------------------------
# Contagem de casos por município
#-----------------------------------------------------------
contagem_municipio = df['municipio'].value_counts()

print(contagem_municipio)
print('--------------------------------------------------')
#-----------------------------------------------------------
# Conntagem total de generos
#-----------------------------------------------------------
contagem_genero = df['genero'].value_counts()

print(contagem_genero)
print('--------------------------------------------------')
#-----------------------------------------------------------
# Contagem total de obitos
#-----------------------------------------------------------
contagem_obitos = df['obito'].value_counts()
total_casos = df['obito'].count()

print(contagem_obitos)
df['obito'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Dados de Óbitos')
plt.ylabel('Quantidade de Óbitos')
plt.xlabel('Evolução para Óbito')
plt.legend([f'Total de Casos: {total_casos}'], loc='upper right')
plt.show()
print('--------------------------------------------------')

#-----------------------------------------------------------
# Contagem de Outliers de casos por estado
#-----------------------------------------------------------


sns.boxplot(contagem_estados, color='skyblue')
plt.title('Outliers com Boxplot')
plt.xlabel('Estados')
plt.ylabel('Quantidade de Casos')

outliers = ['MG, SP, RJ e ES']

outlier_marca = plt.Line2D([], [], 
                           linestyle='none', 
                           marker='o', 
                           color='black', 
                           markerfacecolor='white')
plt.legend([outlier_marca], outliers, title='Outliers', loc='upper right')

plt.show()


#-----------------------------------------------------------
# Gráfico de vacinados por ano
#-----------------------------------------------------------

# converter a coluna 'total_2' para float
df_2['total_2'] = df_2['total_2'].str.replace(',', '.').astype(float) * 1000 

df_vacinados_por_ano = df_2[['Ano', 'total_2']].copy()
df_vacinados_por_ano.columns = ['data_ano', 'quantidade']

# Ordenando o DataFrame pelo ano
df_vacinados_por_ano.sort_values(by='data_ano', inplace=True)

# Exibindo o DataFrame resultante
print(df_vacinados_por_ano)
print('--------------------------------------------------')
plt.figure(figsize=(10, 6))
plt.bar(df_2['Ano'] , df_2['total_2'], color='skyblue')
plt.xlabel('Ano')
plt.ylabel('Total de Vacinados(em milhares)')
plt.title('Total de Vacinados por Ano')
plt.xticks(df_2['Ano'], rotation=45)
plt.grid(True)
plt.show()

#-----------------------------------------------------------
# Fazendo contagem de caso por ano e criando uma novo data frame para usar na regressão linear
#-----------------------------------------------------------

contagem_anos = df['dt_ano'].value_counts().reset_index()

df_contagem_anos = contagem_anos
df_contagem_anos.columns = ['data_ano', 'quantidade']
df_contagem_anos.sort_values(by='data_ano', inplace=True)
print(df_contagem_anos)
print('--------------------------------------------------')


#-----------------------------------------------------------
# Excluindo as linhas dos df_vacinados_por_ano e d_contagem_anos, para analisar apenas os anos compativeis entre elas.
#-----------------------------------------------------------

df_vacinados_por_ano.drop(index=0, inplace=True)
df_vacinados_por_ano.drop(index=18, inplace=True)
df_vacinados_por_ano.drop(index=29, inplace=True)
df_contagem_anos.drop(index=17, inplace=True)
df_contagem_anos.drop(index=28, inplace=True)
print(df_contagem_anos) # imprimindo para verificar se a linha foi removida
print('--------------------------------------------------')
print(df_vacinados_por_ano)# imprimindo para verificar se a linha foi removida

#-----------------------------------------------------------
# Regressão linear
#-----------------------------------------------------------


x = df_vacinados_por_ano['quantidade']
y = df_contagem_anos['quantidade']

coeficiente_angular, intercepto = np.polyfit(x, y, 1)

print(f"Valor coeficiente angular: {coeficiente_angular}")
print(f"Valor intercepto: {intercepto}")

y_pred = coeficiente_angular * x + intercepto
print(y_pred)
print('--------------------------------------------------')
plt.scatter(x, y, color='skyblue',  label="Dados Originais")
plt.plot(x, y_pred, color="red", linewidth=2, label="Linha de Regressão")
plt.xlabel('Quandidade de Vacinados')
plt.ylabel('Quantidade de Casos')
plt.legend()
plt.title('Regressão Linear de Quantidade de Casos de Febre Amarela por Vacinados')
plt.grid(True)
plt.show()