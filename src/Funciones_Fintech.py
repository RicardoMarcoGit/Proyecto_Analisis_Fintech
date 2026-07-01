#!/usr/bin/env python
# coding: utf-8

# # Funciones analisis Fintech

# ## Importamos librerias

# In[1]:


# Librerías estándar
import os
import sys
import warnings
from pathlib import Path

# Manipulación de datos
import pandas as pd
import numpy as np

# Configuración de warnings
warnings.filterwarnings('ignore')

# Análisis de nulos
import missingno as msno

# Visualización de datos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

# Textos
import unicodedata
from fuzzywuzzy import process
import re

# Estadistica
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.proportion import proportions_ztest



# ## Funciones de lectura de archivos

# In[2]:


# Funcion carga de archivos.
def carga_archivos(path):
    # Lista de los archivos en el directorio.
    lista = os.listdir(path)
    print(f"📂 Total archivos encontrados: {len(lista)}")
    print(lista)
    print('*' * 80)

    # Filtrado archivos .csv y .xlsx (CORREGIDO: usar 'or' en lugar de '|')
    files = [file for file in lista if file.endswith('.csv') or file.endswith('.xlsx')]

    print(f"✅ Archivos CSV/Excel válidos: {len(files)}")
    for i, f in enumerate(files, 1):
        print(f"   {i}. {f}")
    print('*' * 80)
    return files



# In[3]:


def leer_archivos(ruta_completa):
    try:
        # Convertimos siempre a Path (robusto y multiplataforma)
        ruta_completa = Path(ruta_completa)

        # Extraemos la extensión de forma segura
        extension = ruta_completa.suffix.lower()

        if extension == '.csv':
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

            for encoding in encodings:
                try:
                    df = pd.read_csv(ruta_completa, sep=None, engine='python', encoding=encoding)
                    print(f"   ✓ Codificación exitosa: {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue

            print(f"   ⚠️ No se pudo decodificar con codificaciones estándar")
            return None

        elif extension in ('.xlsx', '.xls'):
            df = pd.read_excel(ruta_completa)
            return df

        else:
            print(f"   ⚠️ Formato no compatible: {extension}")
            return None

    except FileNotFoundError:
        print(f"   ❌ Archivo no encontrado: '{ruta_completa}'")
        return None

    except Exception as e:
        print(f"   ❌ Error inesperado: {type(e).__name__}: {e}")
        return None


# ## Funciones exploracion inicial

# In[4]:


# Funcion exploracion inicial de datos.
def exploracion_datos(df):
    print('Exploración inicial de datos:')
    print('*'*100)

    # Información general del dataframe.
    num_filas, num_columnas = df.shape
    print(f'El numero de filas es: {num_filas}\nEl numero de columnas es: {num_columnas}')
    print('*'*100)

    # Exploracion visulal de las primeras, últimas y aleatorias filas del dataframe.
    print('Las 5 primeras filas del dataframe son:')
    display(df.head())
    print('*'*100)
    print('Las 5 últimas filas del dataframe son:')
    display(df.tail())
    print('*'*100)
    print('Muestra aleatoria de 5 filas del dataframe:')
    display(df.sample(5))
    print('*'*100)

    # Estadisticos descriptivos del dataframe.
    print('Estadísticos descriptivos del dataframe:')
    display(df.describe())
    print('*'*100)

    # Resumen de tipologia de datos, visualizacion de nulos y valores unicos.
    print('Resumen de tipología de datos, visualización de nulos y valores únicos:')
    df_tipos=df.dtypes.to_frame(name='Tipos de datos')
    df_nulos=df.isnull().sum().to_frame(name='Nulos')
    df_porc_nulos = (df.isnull().sum() / len(df) * 100).to_frame(name='Porcentaje Nulos')
    df_valores_unicos = pd.DataFrame(df.apply(lambda x: x.unique()))
    df_valores_nunicos = pd.DataFrame(df.apply(lambda x: x.nunique()))
    df_por_valores_nunicos=pd.DataFrame(df.apply(lambda x: x.nunique())/df.shape[0]*100)
    df_valores_unicos.rename(columns={0:'Valores unicos'}, inplace=True)
    df_valores_nunicos.rename(columns={0:'Numero valores unicos'}, inplace=True)
    df_por_valores_nunicos.rename(columns={0:'Porcentaje valores unicos'}, inplace=True)
    df_exploracion = pd.concat([df_tipos, df_nulos, df_porc_nulos,df_valores_nunicos,df_por_valores_nunicos,df_valores_unicos], axis=1)

    # MOSTRAR el resumen final
    display(df_exploracion)
    print('*'*100)

    return df_exploracion


# ## Funciones limpieza y preparacion de datos

# In[5]:


# Funcion para renombrar columnas originales.
def renombrar_columnas(df):
    nombre_columnas = ['age', 'job', 'marital_status', 'education', 'credit_default', 'housing_loan', 'personal_loan',
       'contact_type', 'last_contact_month', 'last_contact_day', 'last_contact_duration_secs', 
       'number_contacts', 'number_days_last_contact','number_of_previous_contacts', 
       'outcome_previous_campaign', 'employement_variation_rate', 'consumer_price_index','consumer_confidence_index', 
       'euribor_3m', 'number_employees', 'subscribed_term_deposit', 'age_group',
       'last_contact_duration_mins', 'last_contact_duration_mins_group',
       'employement_variation_rate_group','consumer_price_index_group','consumer_confidence_index_group','euribor_3m_group']
    df.columns = nombre_columnas

    return df


# In[6]:


# Funcion para normalizar textos.
def normalizar_textos(df):
    for col in df.select_dtypes(include=['object']).columns: # Iteramos sobre columnas de tipo object.
        df[col]=df[col].str.lower() # Convertimos a minúsculas.
        df[col]=df[col].str.strip() # Eliminamos espacios en blanco al inicio y final.
        df[col]=df[col].str.replace(r'\.+$', '', regex=True) # Eliminamos los '.' al final de las cadenas de texto.
        df[col]=df[col].str.replace(r'(?<=\w)\.+(?=\w)', '_', regex=True) # Reemplazamos los '.' entre palabras por '_'.
        df[col]=df[col].str.replace(r'(?<=\w)-+(?=\w)', '_', regex=True) # Reemplazamos los '-' entre palabras por '_'.
        dict_job={'admin':'administrative_staff'}
        if col=='job':
            df[col]=df[col].replace(dict_job)

    return df


# In[7]:


# Funcion convertir columnas a tipo categórico.
def columnas_categoricas(df):
    # Listas categorias.
    lista_categorias_job=['unknown','unemployed','student','retired','housemaid', 'services','blue_collar',
                        'self_employed',  'administrative_staff',  'technician','entrepreneur','management']
    lista_marital_status=['unknown','single','married','divorced']
    lista_categorias_education=['unknown','illiterate','basic_4y','basic_6y', 'basic_9y', 
                                    'high_school', 'professional_course',  'university_degree']
    lista_credit_default=['unknown','yes','no']
    lista_housing_loan=['unknown','yes','no']
    lista_personal_loan=['unknown','yes','no']
    lista_contact_type=['telephone','cellular']
    lista_last_contact_month=['mar', 'apr','may', 'jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
    lista_last_contact_day=['mon', 'tue', 'wed', 'thu', 'fri']
    lista_outcome_previous_campaign=['nonexistent', 'failure', 'success']
    lista_age_group=['0 - 25', '26 - 35', '36 - 45', '46 - 55', '56 - 65', '65+']
    lista_last_contact_duration_mins=['0 - 1','1 - 2','2 - 3','3 - 5','5 - 10','10 - 20','20+']
    lista_employement_variation_rate_group=['< -0.5','-0.5 - 0','0 - 0.5','> 0.5']
    lista_consumer_price_index_group=['<90','90 - 94','94 - 100','100 - 104','104+']
    lista_consumer_confidence_index_group=['<=-35', '-35 - 0', '0 - 35','35 - 100','> 100']
    lista_euribor_3m_group=['<0','0 - 1','1 - 2','2 - 3','3 - 4','4 - 5','5+']

    # Convertir columnas a tipo categórico segun las listas definidas.
    df['job'] = pd.Categorical(df['job'], categories=lista_categorias_job, ordered=False)
    df['marital_status'] = pd.Categorical(df['marital_status'], categories=lista_marital_status, ordered=False)
    df['education'] = pd.Categorical(df['education'], categories=lista_categorias_education, ordered=False)
    df['credit_default'] = pd.Categorical(df['credit_default'], categories=lista_credit_default, ordered=False)
    df['housing_loan'] = pd.Categorical(df['housing_loan'], categories=lista_housing_loan, ordered=False)
    df['personal_loan'] = pd.Categorical(df['personal_loan'], categories=lista_personal_loan, ordered=False)
    df['contact_type'] = pd.Categorical(df['contact_type'], categories=lista_contact_type, ordered=False)
    df['last_contact_month'] = pd.Categorical(df['last_contact_month'], categories=lista_last_contact_month, ordered=True)
    df['last_contact_day'] = pd.Categorical(df['last_contact_day'], categories=lista_last_contact_day, ordered=False)
    df['outcome_previous_campaign'] = pd.Categorical(df['outcome_previous_campaign'], categories=lista_outcome_previous_campaign, ordered=False)
    df['age_group']=pd.Categorical(df['age_group'],categories=lista_age_group,ordered=True)
    df['last_contact_duration_mins_group']=pd.Categorical(df['last_contact_duration_mins_group'],categories=lista_last_contact_duration_mins,ordered=True)
    df['employement_variation_rate_group']=pd.Categorical(df['employement_variation_rate_group'],categories=lista_employement_variation_rate_group,ordered=True)
    df['consumer_price_index_group']=pd.Categorical(df['consumer_price_index_group'],categories=lista_consumer_price_index_group,ordered=True)
    df['consumer_confidence_index_group']=pd.Categorical(df['consumer_confidence_index_group'],categories=lista_consumer_confidence_index_group,ordered=True)
    df['euribor_3m_group']=pd.Categorical(df['euribor_3m_group'],categories=lista_euribor_3m_group,ordered=True)

    return df


# In[8]:


# Funcion cambiar columna subscribed_term_deposit a binario.
def normalizar_binario(df):
    dict_subscribed_term_deposit={'yes':1,'no':0} #Creamos un dicionario.
    nulos=set(df.subscribed_term_deposit)-set(dict_subscribed_term_deposit.keys()) # Comprobamos si hay valores no contemplados en el diccionario.
    if nulos:
        print(f'Existen valores nulos en la columna subscribed_term_deposit: {nulos}') # Si hay valores no contemplados, los mostramos.
    df.subscribed_term_deposit=df.subscribed_term_deposit.replace(dict_subscribed_term_deposit)
    df.subscribed_term_deposit = df.subscribed_term_deposit.astype('int64') # Convertimos a int64.


# ## Funciones graficos y obtencion de datos

# In[9]:


def cramers_v(df):

    chi2, p, dof, expected = chi2_contingency(df)
    n = df.values.sum()              
    k = min(df.shape) - 1            

    return np.sqrt(chi2 / (n * k))


# In[10]:


def heatmap_correlation(df, df_2, columna, title, ylable):

    # Calculamos chi2, p, dof
    chi2, p, dof, expected = stats.chi2_contingency(df)

    # Calculamos V de Cramer
    V = cramers_v(df)

    # Creamos la figura.
    plt.figure(figsize=(10, 6))

    # Normalizamos los datos para obtener el porcentaje sobre el total de filas.
    df_heatmap = df. div(df.sum(axis=1), axis=0)

    # Creamos heatmap
    ax = sns.heatmap(
        df_heatmap,
        annot=True,
        fmt='.2%',
        cmap='YlGnBu',
        cbar_kws={'label': 'Porcentaje', 'format': PercentFormatter(1)},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel(ylable, fontsize=12)

    # Conteos y porcentajes
    df_values = df_2[columna]. value_counts().to_frame().transpose()
    df_values_norm = (df_2[columna].value_counts(normalize=True) * 100).to_frame().transpose()

    # Imprimimos resultados
    print(f"chi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}")
    print(f"V de Cramer = {V:.5f}")
    print("*" * 80)
    display(df_values)
    print("*" * 80)
    display(df_values_norm)
    print("*" * 80)
    display(df)

    plt.tight_layout()
    plt.show()


# In[11]:


# Funcion para crear un histograma de barras para la correlacion de Spearman.
def grafico_correlacion_spearman(df,df_2,title):

    # Creamos la figura
    fig, ax = plt.subplots(figsize=(6, 4))

    # Extraemos los datos del dataframe
    x_positions = df.index 
    y_values = df['subscribed_term_deposit'] 

    # Creamos el grafico de barras
    bars = ax.bar(
        x=x_positions,      
        height=y_values,   
        color="coral",
        edgecolor="black",
        alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Correlación", fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Creamos un conteo de los indices del dataframe a graficar sobre el dataframe original
    for col in df.index:
        df_values=df_2[col].value_counts().to_frame().transpose()
        df_values_norm=(df_2[col].value_counts(normalize=True)*100).to_frame().transpose()
        display(df_values)
        display(df_values_norm)
        print("*" * 80)
    display(df)
    print("*" * 80)
    plt.tight_layout()
    plt.show()


# In[12]:


# Funcion para realizar un bucle de analisis de dataframe.
def bucle_analisis_dataframe(df,df_analizar,columnas):

    for col in columnas:
        df_temp = (
            df_analizar
            .groupby(col)
            .size()
            .reset_index(name='count'))

        df_temp['percentage'] = df_temp['count'] / df_temp['count'].sum() * 100
        df_temp = df_temp[df_temp['count'] > 0]
        df_temp = df_temp.sort_values(by='count', ascending=False)
        df_temp = df_temp.set_index(col)
        df[col]=df_temp


# In[13]:


# Funcion para realizar un bucle de analisis de dataframe con graficos 3 plot.
def analisis_subscribed_term_deposit_3plot(df,col1,col2,col3):
    # Graficamos las variables
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Gráfico col1
    axes[0].bar(
        df[col1].index.astype(str),
        df[col1]['percentage'], 
        color='steelblue',
        edgecolor='black',
        alpha=0.7)
    axes[0].set_title(f'Distribución de {col1} con deposito contratado', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('%')
    axes[0].tick_params(axis='x', rotation=0, labelsize=10)
    axes[0].grid(axis='y', alpha=0.3)

    # Gráfico col2
    axes[1].bar(
        df[col2].index.astype(str),
        df[col2]['percentage'],
        color='coral',
        edgecolor='black',
        alpha=0.7)
    axes[1].set_title(f'Distribución de {col2} con deposito contratado', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('%')
    axes[1].tick_params(axis='x', rotation=45)
    plt.setp(axes[1].get_xticklabels(), ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    # Gráfico col3
    axes[2].bar(
        df[col3].index.astype(str),
        df[col3]['percentage'],
        color='mediumseagreen',
        edgecolor='black',
        alpha=0.7)
    axes[2].set_title(f'Distribución de {col3} con deposito contratado', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('%')
    axes[2].grid(axis='y', alpha=0.3)


    display(df[col1].T)
    print('*'*100)
    display(df[col2].T)
    print('*'*100)
    display(df[col3].T)
    print('*'*100)
    plt.tight_layout()
    plt.show()


# In[14]:


# Funcion para realizar un bucle de analisis de dataframe con graficos 4 plot.
def analisis_subscribed_term_deposit_4plot(df,col1,col2,col3,col4):
    # Graficamos las variables
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Gráfico col1
    axes[0, 0].bar(
        df[col1].index.astype(str),
        df[col1]['percentage'], 
        color='steelblue',
        edgecolor='black',
        alpha=0.7)
    axes[0, 0].set_title(f'Distribución de {col1} con deposito contratado', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('%')
    axes[0, 0].tick_params(axis='x', rotation=0, labelsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Gráfico col2
    axes[0, 1].bar(
        df[col2].index.astype(str),
        df[col2]['percentage'],
        color='coral',
        edgecolor='black',
        alpha=0.7)
    axes[0, 1].set_title(f'Distribución de {col2} con deposito contratado', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].tick_params(axis='x', rotation=45)
    plt.setp(axes[0, 1].get_xticklabels(), ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Gráfico col3
    axes[1, 0].bar(
        df[col3].index.astype(str),
        df[col3]['percentage'],
        color='mediumseagreen',
        edgecolor='black',
        alpha=0.7)
    axes[1, 0].set_title(f'Distribución de {col3} con deposito contratado', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('%')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Gráfico col4
    axes[1, 1].bar(
        df[col4].index.astype(str),
        df[col4]['percentage'],
        color='orange',
        edgecolor='black',
        alpha=0.7)
    axes[1, 1].set_title(f'Distribución de {col4} con deposito contratado', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.setp(axes[1, 1].get_xticklabels(), ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    display(df[col1].T)
    print('*'*100)
    display(df[col2].T)
    print('*'*100)
    display(df[col3].T)
    print('*'*100)
    display(df[col4].T)
    print('*'*100)
    plt.tight_layout()
    plt.show()


# In[15]:


# Funcion para realizar un bucle de analisis de dataframe con graficos 6 plot.
def analisis_subscribed_term_deposit_6plot(df,col1,col2,col3,col4,col5,col6):
    # Graficamos las variables
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Gráfico col1
    axes[0, 0].bar(
        df[col1].index.astype(str),
        df[col1]['percentage'], 
        color='steelblue',
        edgecolor='black',
        alpha=0.7)
    axes[0, 0].set_title(f'Dist. {col1} dep-contratado', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('%')
    axes[0, 0].tick_params(axis='x', rotation=0, labelsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Gráfico col2
    axes[0, 1].bar(
        df[col2].index.astype(str),
        df[col2]['percentage'],
        color='coral',
        edgecolor='black',
        alpha=0.7)
    axes[0, 1].set_title(f'Dist. {col2} dep-contratado', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].tick_params(axis='x', rotation=45)
    plt.setp(axes[0, 1].get_xticklabels(), ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Gráfico col3
    axes[1, 0].bar(
        df[col3].index.astype(str),
        df[col3]['percentage'],
        color='mediumseagreen',
        edgecolor='black',
        alpha=0.7)
    axes[1, 0].set_title(f'Dist. {col3} dep-contratado', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('%')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Gráfico col4
    axes[1, 1].bar(
        df[col4].index.astype(str),
        df[col4]['percentage'],
        color='orange',
        edgecolor='black',
        alpha=0.7)
    axes[1, 1].set_title(f'Dist. {col4} dep-contratado', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].tick_params(axis='x', rotation=45)
    plt.setp(axes[1, 1].get_xticklabels(), ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Gráfico col5
    axes[0, 2].bar(
        df[col5].index.astype(str),
        df[col5]['percentage'],
        color='mediumpurple',
        edgecolor='black',
        alpha=0.7)
    axes[0, 2].set_title(f'Dist. {col5} dep-contratado', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('%')
    axes[0, 2].tick_params(axis='x', rotation=45)
    plt.setp(axes[0, 2].get_xticklabels(), ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)

    # Gráfico col6
    axes[1, 2].bar(
        df[col6].index.astype(str),
        df[col6]['percentage'],
        color='gold',
        edgecolor='black',
        alpha=0.7)
    axes[1, 2].set_title(f'Dist. {col6} dep-contratado', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('%')
    axes[1, 2].tick_params(axis='x', rotation=45)
    plt.setp(axes[1, 2].get_xticklabels(), ha='right')
    axes[1, 2].grid(axis='y', alpha=0.3)

    display(df[col1].T)
    print('*'*100)
    display(df[col2].T)
    print('*'*100)
    display(df[col3].T)
    print('*'*100)
    display(df[col4].T)
    print('*'*100)
    display(df[col5].T)
    print('*'*100)
    display(df[col6].T)
    print('*'*100)
    plt.show()

