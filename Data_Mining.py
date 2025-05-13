import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import mean
from numpy import std

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from funcoes import build_target, mode_high_frequencie, clip_tail, scaler


df = pd.read_csv(r"C:\Users\schools.csv")

#filtando pelas escolas da Bahia
df = df.loc[(df["CO_UF"]==29)]

#filtando pelos concluintes
conclusao = df.loc[(df["TP_ST_CONCLUSAO"]==2)]

#filtando todas as escolas estaduais
adm = conclusao.loc[(conclusao["CO_DEPENDENCIA_ADM"]==2)]

#filtrando as escolas com energia, água e esgoto
agua = adm.loc[(adm["IN_AGUA_INEXISTENTE"]==0)]
esgoto = agua.loc[(agua["IN_ESGOTO_INEXISTENTE"]==0)]
energia = esgoto.loc[(esgoto["IN_ENERGIA_INEXISTENTE"]==0)]

#filtrando apenas escolas com mais de 10 alunos
df = energia.loc[(energia["QT_MATRICULAS"]>=10)]

cols = ["CO_ANO", "EDU_PAI", "EDU_MAE", "IN_LABORATORIO_INFORMATICA",
        "IN_LABORATORIO_CIENCIAS", "IN_SALA_ATENDIMENTO_ESPECIAL", "IN_BIBLIOTECA", "IN_SALA_LEITURA",
        "IN_BANHEIRO", "IN_BANHEIRO_PNE", "QT_SALAS_UTILIZADAS", "QT_EQUIP_TV", "QT_EQUIP_DVD",
        "QT_EQUIP_COPIADORA", "QT_EQUIP_IMPRESSORA", "QT_COMP_ALUNO", "IN_BANDA_LARGA", "QT_FUNCIONARIOS",
        "IN_ALIMENTACAO", "IN_SALA_PROFESSOR", "IN_QUADRA_ESPORTES", "IN_COZINHA", "IN_EQUIP_PARABOLICA",
        "IN_ATIV_COMPLEMENTAR", "TITULACAO", "NU_CIENCIA_NATUREZA", "NU_CIENCIAS_HUMANAS",
        "NU_LINGUAGENS_CODIGOS", "NU_MATEMATICA", "NU_ESCOLAS", "NU_LICENCIADOS", "IN_FORM_DOCENTE", "RENDA_PERCAPITA",
        "DIVERSIDADE", "TP_COR_RACA_0.0", "TP_COR_RACA_1.0", "TP_COR_RACA_2.0", "TP_COR_RACA_3.0", "TP_COR_RACA_4.0",
        "TP_COR_RACA_5.0", "TP_SEXO", "NU_NOTA_GERAL"]

df = df[cols]

#binarizando valores de variaveis numericas 
QT_to_IN = ['QT_EQUIP_DVD', 'QT_EQUIP_COPIADORA', 'QT_EQUIP_IMPRESSORA', 'QT_EQUIP_TV']
conds = [df[QT_to_IN].values == 0 , df[QT_to_IN].values > 0]
choices = [0, 1]
df[QT_to_IN] = pd.DataFrame(np.select(conds, choices), index=df[QT_to_IN].index, columns=df[QT_to_IN].columns)

#construindo a coluna TARGET
df_2009 = df[df["CO_ANO"] == 2009]
df_2010 = df[df["CO_ANO"] == 2010]
df_2011 = df[df["CO_ANO"] == 2011]
df_2012 = df[df["CO_ANO"] == 2012]
df_2013 = df[df["CO_ANO"] == 2013]
df_2014 = df[df["CO_ANO"] == 2014]
df_2015 = df[df["CO_ANO"] == 2015]
df_2016 = df[df["CO_ANO"] == 2016]
df_2017 = df[df["CO_ANO"] == 2017]
df_2018 = df[df["CO_ANO"] == 2018]
df_2019 = df[df["CO_ANO"] == 2019]
ano = [df_2009, df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017, df_2018, df_2019]
for i in ano:
  build_target(i)


#limpando dataframes por ano
years = [df.drop(['NU_NOTA_GERAL', 'CO_ANO'], axis=1) for df in [df_2009, df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016, df_2017, df_2018, df_2019]]

#condicao de cross-validacao
cv = StratifiedKFold(n_splits=5)

#Regressao Logistica
results_df = []
features_df = pd.DataFrame()
n = 2009

for i in years:
  remover_colunas = mode_high_frequencie(i)
  remover_colunas.append('TARGET')
  i = clip_tail(i)
  X = i.drop(remover_colunas, axis=1)
  y = i['TARGET']
  X = scaler(X)
  model = LogisticRegression()
  auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
  mean_auc = auc_scores.mean()
  results_df.append([n, mean_auc])
  n = n + 1
  model.fit(X, y)
  coefficients = model.coef_[0]
  feature_importance = pd.DataFrame({'Feature': X.columns, 'Importancia': np.abs(coefficients)})
  features_df = pd.concat([features_df, feature_importance], ignore_index=True)
#resultados AUC Regressao Logistica
results_lr = pd.DataFrame(results_df, columns = ['Ano', 'AUC'])


#Random Forest
results_df1 = []
features_df1 = pd.DataFrame()
n = 2009

for i in years:
  remover_colunas = mode_high_frequencie(i)
  remover_colunas.append('TARGET')
  i = clip_tail(i)
  X = i.drop(remover_colunas, axis=1)
  y = i['TARGET']
  X = scaler(X)
  rf = RandomForestClassifier()
  auc_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
  mean_auc = auc_scores.mean()
  results_df1.append([n, mean_auc])
  n = n + 1
  rf.fit(X, y)
  feature_scores = pd.DataFrame({'Feature': X.columns, 'Importancia': rf.feature_importances_})
  features_df1 = pd.concat([features_df1, feature_scores], ignore_index=True)
#resultados AUC Random Forest
results_rf = pd.DataFrame(results_df1, columns = ['Ano', 'AUC'])

#print(results_lr, results_rf)

#plotando grafico AUC
log_reg = features_df.groupby("Feature")["Importancia"].sum().sort_values(ascending=True)
rand_for = features_df1.groupby("Feature")["Importancia"].sum().sort_values(ascending=True)

df_auc = pd.DataFrame({'AUC_LR': results_lr['AUC'],
                   'AUC_RF': results_rf['AUC']})
df_auc = df_auc.set_index(results_lr['Ano'])

pal = sns.color_palette("husl", 8)
sns.lineplot(data=df_auc, markers=True, palette=pal * df_auc.columns.size)
plt.show()


#renomeando features
rename_dict = {"QT_EQUIP_TV": "Quantidade de Equipamentos de TV",
                  "IN_ALIMENTACAO": "Alimentacao",
                  "IN_SALA_ATENDIMENTO_ESPECIAL": "Sala de Atendimento Especial",
                  "IN_LABORATORIO_INFORMATICA": "Laboratório de Informática",
                  "QT_EQUIP_DVD": "Quantidade de Equipamentos de DVD",
                  "IN_ATIV_COMPLEMENTAR": "Atividade Complementar",
                  "IN_COZINHA": "Cozinha",
                  "IN_QUADRA_ESPORTES": "Quadra de Esportes",
                  "IN_SALA_LEITURA": "Sala de Leitura",
                  "IN_BIBLIOTECA": "Biblioteca",
                  "IN_EQUIP_PARABOLICA": "Equipamento de Antena Parabólica",
                  "IN_BANHEIRO_PNE":"Banheiro PNE",
                  "IN_BANDA_LARGA": "Internet Banda Larga",
                  "QT_EQUIP_COPIADORA": "Quantidade de Equipamento Copiadora",
                  "TP_COR_RACA_2.0": "Raça Preta",
                  "TP_COR_RACA_3.0": "Raça Parda",
                  "TP_COR_RACA_1.0": "Raça Branca",
                  "TP_COR_RACA_4.0": "Raça Amarela",
                  "TP_COR_RACA_5.0": "Raça Indígena",
                  "TP_COR_RACA_0.0": "Raça Não Declarada",
                  "IN_LABORATORIO_CIENCIAS": "Laboratorio de Ciências",
                  "QT_COMP_ALUNO": "Quantidade de Computador",
                  "TP_SEXO": "Sexo",
                  "NU_CIENCIA_NATUREZA": "Docentes Ciências da Natureza",
                  "NU_MATEMATICA": "Docentes Matemática",
                  "NU_CIENCIAS_HUMANAS": "Docentes Ciências Humanas",
                  "NU_LINGUAGENS_CODIGOS": "Docentes Linguagens e Códigos",
                  "NU_LICENCIADOS": "Docentes com Formação Pedagógica",
                  "QT_SALAS_UTILIZADAS": "Quantidade de Salas Utilizadas",
                  "NU_ESCOLAS": "Empregos dos Docentes",
                  "DIVERSIDADE": "Diversidade",
                  "QT_FUNCIONARIOS": "Quantidade de Funcionários",
                  "IN_FORM_DOCENTE": "Formação Docente",
                  "TITULACAO": "Titulação do Docente",
                  "EDU_PAI": "Educação do Pai",
                  "EDU_MAE": "Educação da Mãe",
                  "RENDA_PERCAPITA": "Renda Per Capita"}

log_reg = log_reg.rename(index=rename_dict)
rand_for = rand_for.rename(index=rename_dict)

#plotando grafico comparacao importancia das features
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(13,13))
log_reg.plot(x='Feature', y='Importancia', kind='barh', ax = axes[0], color=plt.cm.Set2(log_reg / max(log_reg)))
rand_for.plot(x='Feature', y='Importancia', kind='barh', ax = axes[1], color=plt.cm.Set2(rand_for / max(rand_for)))

axes[0].set_title("Regressão Logística", fontsize=16)
axes[1].set_title("Floresta Aleatória", fontsize=16)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                    top=0.9, wspace=0.90)
plt.show()