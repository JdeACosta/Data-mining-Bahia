import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#funcao de construir feature TARGET
def build_target(df):
  df['TARGET'] = pd.qcut (df.NU_NOTA_GERAL, 4, labels = [1,2,3,4]).map(lambda x : 0 if x!=4 else 1)
  return df['TARGET']


#funcao de eliminar variaveis pela moda
def mode_high_frequencie(df):
    columns_dropped = []
    ammount=0
    before= df.shape[1]
    for i in df:
        mode = df[i].mode()[0]
        threshold = 0.9
        count = df[(df[i]== mode)].shape[0]
        freq = count/df.shape[0]
        if freq >= threshold:
            ammount +=1
            columns_dropped.append(i)

    return(columns_dropped)


#funcao de winsorizacao
def clip_tail(df):
  quantitative = df[(df.nunique() > 2).index[(df.nunique() > 2)]].columns.to_list()
  #print(quantitative)
  df[quantitative]=df[quantitative].apply(lambda x: x.clip(upper = (np.percentile(x, 97.5))))
  df[quantitative]=df[quantitative].apply(lambda x: x.clip(lower = (np.percentile(x, 2.5))))
  return(df)


#funcao de padronizacao
def scaler(df):
  scaler = MinMaxScaler()
  x = df.values
  x_scaled = scaler.fit_transform(x)
  df = pd.DataFrame(x_scaled, columns = df.columns)
  return (df)