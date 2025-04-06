import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cols_aa = ["CO_ANO", "CO_ESCOLA", "TP_SEXO", "TP_COR_RACA", "TP_ST_CONCLUSAO",
        "NU_NOTA_GERAL","CO_UF", "CO_DEPENDENCIA_ADM"]

df_aa = pd.read_csv(r"C:\Users\students__aa.csv", usecols = cols_aa)

cols = [1,2,4,5,6,16,18,19]

df_ab = pd.read_csv(r"C:\Users\students__ab.csv", header=None, usecols = cols)
df_ac = pd.read_csv(r"C:\Users\students__ac.csv", header=None, usecols = cols)
df_ad = pd.read_csv(r"C:\Users\students__ad.csv", header=None, usecols = cols)
df_ae = pd.read_csv(r"C:\Users\students__ae.csv", header=None, usecols = cols)
df_af = pd.read_csv(r"C:\Users\students__af.csv", header=None, usecols = cols)
df_ag = pd.read_csv(r"C:\Users\students__ag.csv", header=None, usecols = cols)
df_ah = pd.read_csv(r"C:\Users\students__ah.csv", header=None, usecols = cols)
df_ai = pd.read_csv(r"C:\Users\students__ai.csv", header=None, usecols = cols)
df_aj = pd.read_csv(r"C:\Users\students__aj.csv", header=None, usecols = cols)
df_ak = pd.read_csv(r"C:\Users\students__ak.csv", header=None, usecols = cols)

df_concat = pd.concat([df_ab, df_ac, df_ad, df_ae, df_af, df_ag, df_ah, df_ai, df_aj, df_ak])

df_concat.columns = (["CO_ANO", "CO_ESCOLA", "TP_SEXO", "TP_COR_RACA", "TP_ST_CONCLUSAO",
        "NU_NOTA_GERAL","CO_UF", "CO_DEPENDENCIA_ADM"])

#concatenando todos os dataframes
df_students = pd.concat([df_aa, df_concat])

#filtando pelos concluintes
conclusao = df_students.loc[(df_students["TP_ST_CONCLUSAO"]==2)]

#filtando todas as escolas estaduais
adm = conclusao.loc[(conclusao["CO_DEPENDENCIA_ADM"]==2)]

#filtrando apenas as escolas da Bahia
ba_df = adm.loc[(adm["CO_UF"]==29)]

#informacoes das notas por genero (0: fem; 1: masc)
ba_df['TP_SEXO'] = ba_df['TP_SEXO'].map({0.0: 'Feminino', 1.0: 'Masculino'})
ba_gn_geral = ba_df[["CO_ANO", "TP_SEXO","NU_NOTA_GERAL"]].groupby(['CO_ANO', 'TP_SEXO']).describe()
print(ba_gn_geral)

#grafico de desempenho por genero
palette = sns.color_palette("Set2")
fig, axes = plt.subplots(figsize=(20,12))
sns.boxplot(data=ba_df, x="CO_ANO", y="NU_NOTA_GERAL", hue="TP_SEXO", palette=palette) #Gênero: 0 = feminino, 1 = masculino
axes.set(xlabel="ANO", ylabel="NOTA GERAL")
sns.move_legend(axes, title=None, loc='best')
sns.set_theme(font_scale=1.5, style="whitegrid")
plt.show()

#informacoes das notas por raca (0: Não Declarado, 1: Branca, 2: Preta, 3: Parda, 4: Amarela, 5: Indígena)
ba_df['TP_COR_RACA'] = ba_df['TP_COR_RACA'].map({0.0: 'Não Declarado', 1.0: 'Branca', 2.0: 'Preta', 3.0: 'Parda', 4.0: 'Amarela', 5.0: 'Indígena'})
ba_rc_geral = ba_df[["CO_ANO", "TP_COR_RACA","NU_NOTA_GERAL"]].groupby(['CO_ANO', 'TP_COR_RACA']).describe()
print(ba_rc_geral)

#grafico de desempenho por raca
fig, axes = plt.subplots(figsize=(20,12))
palette = sns.color_palette("pastel")
sns.boxplot(data=ba_df, x="CO_ANO", y="NU_NOTA_GERAL", hue="TP_COR_RACA", palette=palette)
axes.set_xlabel("ANO", fontsize=16)
axes.set_ylabel("NOTA GERAL", fontsize=16)
sns.move_legend(axes, title=None, loc='lower right')
sns.set_theme(font_scale=1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.set_theme(style="whitegrid")
plt.show()