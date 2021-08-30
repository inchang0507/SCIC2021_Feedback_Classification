import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN,SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

class Preprocessor():
    def __init__(self):
        print("Preprocessor is Operating")

    def preprocessing_4_train(self, dir):
        df = pd.read_excel(dir)
        df = df[['발화','최종분류(우선순위 가장 높은것 선택)']]
        encoder = LabelEncoder()
        encoder.fit(df['최종분류(우선순위 가장 높은것 선택)'],)
        df['label'] = encoder.transform(df['최종분류(우선순위 가장 높은것 선택)'],)
        df = df[['발화','label']].reset_index(drop=True)

        return df, encoder


def make_list (df):
  data_list = []
  for i, label in zip(df['발화'], df['label']):
    data=[]
    data.append(i)
    data.append(label)

    data_list.append(data)
  return data_list


def test_prepro(test):
    test = test.rename(columns={'TEXT':'발화','INT':'label'})
    test['label'] = 30
    test = test[['발화','label']].reset_index(drop=True)

    return test


def oversampling_4_roberta(train):
    noneed_label = []
    for i in dict(train.label.value_counts()).keys():
        if (dict(train.label.value_counts())[i] > 50) or (dict(train.label.value_counts())[i] == 1):
            noneed_label.append(i)

    val_list= []
    val_list2= []

    for i in range(len(train)):
      if train.iloc[i,1] in noneed_label:
        val_list.append(i)
      else:
        val_list2.append(i)

    noneed_over = train.iloc[val_list,:].reset_index(drop=True)
    need_over = train.iloc[val_list2,:].reset_index(drop=True)

    vec = TfidfVectorizer()
    X = vec.fit_transform(need_over['발화'])

    sm = SMOTE(k_neighbors=1)
    lb = LabelBinarizer()
    y_train_bin = lb.fit_transform(need_over['label'])


    X_train_res, y_train_res = sm.fit_sample(X, y_train_bin)

    X_over = vec.inverse_transform(X_train_res)
    y_train_res = lb.inverse_transform(y_train_res)

    X_over_2 = []
    for i in X_over:
        a = ' '.join(i)
        X_over_2.append(a)

    data_list_over = []
    for q, label in zip(X_over_2, y_train_res)  :
        data = []
        data.append(q)
        data.append(label)

        data_list_over.append(data)

    train_4_roberta = pd.concat([pd.DataFrame(data_list_over,columns=['발화','label']),noneed_over],axis=0)
    train_4_roberta['label'] = train_4_roberta['label'].astype('int')    

    return train_4_roberta
