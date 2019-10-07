# -*_ coding: utf-8 -*-
"""
Created on Aug 14 19:06:13 2019
@author: SHAO XUAN HUANG
@email: cyeninesky3@gmail.com

Bairong: Data preprocess, TCN modeling.
"""

from tensorflow import keras
from utils import get_bin_sizes, get_KS_IV

from tcn import compiled_tcn
import tensorflow.python.framework.dtypes
import time
import pickle
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout
from keras.utils import to_categorical
from tqdm import tqdm
from keras.utils import plot_model
from IPython.display import Image
import re

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings("ignore")

def process_effect_dataset(data):
    print('===== Now processing')    
    ranked_feas = list(data.keys()) # get all feature names as a list object
    remove_list = ['flagy','user_date','cus_num','flag_applyloanmon']
    for i in remove_list:
        ranked_feas.remove(i)
    # del ranked_feas[2] # delete 'category' column
    ranked_feas.sort()
    
    IVs={}
    KSs={}
    error_code=[]
    for feat in ranked_feas:
        try:
            sub = data[['flagy', feat]]
            # sub = sub.dropna()
            sub = sub.fillna(-2)
            sub = sub.sort_values(by=[feat]) # order from low to high
            grp_num = get_bin_sizes(sub, feat)
            KS, IV = get_KS_IV(sub, grp_num)
            KSs[feat] = KS
            IVs[feat] = IV
        except:
            error_code.append(feat)
    order_by_value_ks = sorted([(i[1],i[0]) for i in KSs.items()], reverse=True) # 依照 value 排序
    order_by_value_iv = sorted([(i[1],i[0]) for i in IVs.items()], reverse=True) # 依照 value 排序
    return order_by_value_ks, order_by_value_iv, error_code

def process_train(df, iv_top, num_feats, num_classes, test_size = 0.1):
    train_df = df.copy()
    train_df = train_df.fillna(0)
    X = train_df.loc[:, iv_top]
    y = train_df.loc[:,'flagy']
    # 打乱并划分训练集
    train_X, val_X, train_y, val_y = train_test_split(X,y, shuffle=True, test_size=test_size, random_state = 0)

    #TODO: 補歸一化的方式
    maxab_scaler = preprocessing.MaxAbsScaler()
    train_X = maxab_scaler.fit_transform(train_X)
    val_X = maxab_scaler.fit_transform(val_X)
    # train_X = preprocessing.normalize(train_X, norm='l2')
    # val_X = preprocessing.normalize(val_X, norm='l2')

    train_X = train_X.reshape(-1, num_feats, 1)
    val_X = val_X.reshape(-1, num_feats, 1)
    train_y = to_categorical(train_y, num_classes)
    val_y = to_categorical(val_y, num_classes)

    train_X = train_X.astype('float32')
    val_X = val_X.astype('float32')
    print('INTO process_train: ')
    print(train_X.shape, train_y.shape)
    print('='*80)
    return (train_X, train_y), (val_X, val_y)

def _extract_mon(string):
    mon = int(re.findall(r"\d+\.?\d*", string)[0])
    return mon

def _extract_day(m):
    if m == 15 or m==7:
        return float(m)/100
    else:
        return m

def data_preprocess(df, num_feats, num_classes):
    """
    特徵預處理，取iv值 排名 topN 的特徵(代表有區分度)，返回訓練集測試集分別對應的 => (特徵X, 標籤Y)
    :params df: 原始數據
    :params num_feats: 特徵個數
    :params num_classes: 分類個數
    :return: 訓練與測試數據
    """
    
    num = df.isnull().sum()
    too_much_nan_fea_name = list(num[num>13310].index)
    df = df[[fea for fea in list(df.columns) if fea not in too_much_nan_fea_name]]

    order_ks, order_iv, _ = process_effect_dataset(df)
    ks = pd.DataFrame(list(order_ks), columns=['ks', 'name'])
    iv = pd.DataFrame(list(order_iv), columns=['iv', 'name'])

    result= pd.merge(ks, iv, how='left', on='name')
    result = result.sort_values(by=['iv'], ascending=False)
    result.to_csv('./feats_na.csv', index=False)

    # print('order by iv: ')
    # print(result)
    # print('='*80)
    result = result.iloc[0:num_feats,:]
    result['mon'] = result['name'].apply(_extract_mon)
    result['mon'] = result['mon'].apply(_extract_day)
    result = result.sort_values(by=['mon'])    
    iv_top = result.name.tolist()
    
    (train_X, train_y), (val_X, val_y)= process_train(df, iv_top, num_feats, num_classes)
    return (train_X, train_y), (val_X, val_y)

def run_task(epoch, num_feats, num_classes):
    """
    TCN 模型訓練、預測與評估
    :params epoch:  迭代次數
    :params num_feats: 特徵個數
    :params num_classes: 分類個數
    :return: 各項指標, list格式 => [metric1, metric2, ...]
    """

    df = pd.read_csv('./bairong.csv')
    df = df.dropna(axis=1,how='all')

    (x_train, y_train), (x_test, y_test) = data_preprocess(df, num_feats, num_classes)
    model = compiled_tcn(return_sequences=False,
                         num_feat=1,
                         num_classes=2,
                         nb_filters=20,
                         kernel_size=5,
                         dilations=[2 ** i for i in range(10)],
                         nb_stacks=2,
                         max_len=x_train[0:1].shape[1],
                         use_skip_connections=True)

    #---------- Model Visualization ---------#
    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    # Image(filename = 'model.png')

    model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=epoch, verbose=0,
              validation_data=(x_test, y_test.squeeze().argmax(axis=1)))

    #---------- Model Evaluation ------------#
    print('Now start test pred: ')    
    y_pred = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:,1], pos_label=None,sample_weight=None,drop_intermediate=True)
    auc_score = auc(fpr, tpr)
    y_test = [list(i).index(1) for i in y_test]

    y_pred    = list((y_pred>=0.3).astype(int)[:,1])
    f1        = f1_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    ks = max(tpr-fpr)
    
    print('AUC: {}, f1: {}, recall:{}, precision:{}'.format(auc_score, f1, recall, precision))
    print('='*80)
    return [auc_score, fpr, tpr, thresholds, f1, recall, ks, precision]

if __name__ == '__main__':

    #----------- 不同epoch數實驗 ----------#
    results = {}
    num_feat = 100
    num_classes = 2
    with tqdm(total=len(list(range(1, 10, 1)))) as pbar:
        for epoch in range(1, 10, 1):
            # print('Now num feats : {}'.format(epoch))
            [auc_score, fpr, tpr, thresholds, f1, recall, ks, precision] = run_task(epoch, num_feat, num_classes)
            results[epoch] = [auc_score, fpr, tpr, thresholds, f1, recall, ks, precision, epoch]
            print('num feats={}, epoch= {}, auc_score ={}'.format(num_feat, epoch, auc_score))
            pbar.update()
    df = pd.DataFrame.from_dict(results, orient='index', columns=['auc_scores', 'fpr', 'tpr', 'thresholds', 'f1', 'recall', 'ks', 'precision', 'epochs'])
    df.to_csv('./result_2019100301.csv')

    #----------- 不同特徵數實驗 ----------#
    # results={}
    # epoch = 6
    # num_classes =2
    # with tqdm(total=len(list(range(100, 800, 50)))) as pbar:
    #     for num in range(100, 800, 50):
    #         print('Now num feats : {}'.format(num))
    #         [auc_score, fpr, tpr, thresholds, f1, recall, ks, precision] = run_task(epoch, num, num_classes)
    #         results[num] = [auc_score, fpr, tpr, thresholds, f1, recall, ks, precision, epoch]
    #         print('num feats={}, epoch= {}, auc_score ={}'.format(num, epoch, auc_score))
    #         pbar.update()
    # df = pd.DataFrame.from_dict(results, orient='index', columns=['auc_scores', 'fpr', 'tpr', 'thresholds', 'f1', 'recall', 'ks', 'precision', 'epochs'])
    # df.to_csv('./result_2019082102.csv')