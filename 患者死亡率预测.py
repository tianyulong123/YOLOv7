import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 导入数据 加载数据
df=pd.read_excel('D:\ACTA.xls')
# 预览数据类型
df.info()
# 看看数值型变量的描述性统计信息
print(df.describe())
#预览数据
print(df.head())
#查看数据大小
print(df.shape)

# 转化数据类型
# df['treat治疗方式'].apply(convert_currency)
# df["treat治疗方式"] = pd.to_numeric(df["treat治疗方式"],errors='coerce')
# df.dtypes
# df.info()

# 数据预处理
# 1. 查看重复值
print(df.duplicated().sum())
# 去重复
print("去重前数据大小为：{0}".format(df.shape))
df.drop_duplicates()
print("去重后数据大小为：{0}".format(df.shape))

# 2.缺失值处理
print(df.isnull().sum())
import pandas as pd
# 统计缺失值数量
missing=df.isnull().sum().reset_index().rename(columns={0:'missNum'})
# 计算缺失比例
missing['missRate']=missing['missNum']/df.shape[0]
# 按照缺失率排序显示
miss_analy=missing[missing.missRate>0].sort_values(by='missRate',ascending=False)
# miss_analy 存储的是每个变量缺失情况的数据框
# # 缺失值的可视化
# fig = plt.figure(figsize=(18,6)),plt.bar(np.arange(miss_analy.shape[0]), list(miss_analy.missRate.values), align = 'center',color=['red','green','yellow','steelblue'])
# plt.title('Histogram of missing value of variables')
# plt.xlabel('variables names')
# plt.ylabel('missing rate')
# # 添加x轴标签，并旋转90度
# plt.xticks(np.arange(miss_analy.shape[0]),list(miss_analy['index']))
# plt.xticks(rotation=90)
# plt.show()



# 处理缺失值的原则是尽量填充，最后才是删除。
# 缺失值填充的原则：1.分类型数据：众数填充
#                2.数值型数据：正态分布，均值/中位数填充；偏态分布，中位数填充。

#  “中位数”填充
# df.fillna({'X19':df['X19'].median()},inplace=True)
# 对于缺失值占比多的进行均值填充
df.fillna({'X19':df['X19'].mean()},inplace=True)
df.fillna({'X20':df['X20'].mean()},inplace=True)
df.fillna({'X22':df['X22'].mean()},inplace=True)
df.fillna({'X18':df['X18'].mean()},inplace=True)
df.fillna({'X16':df['X16'].mean()},inplace=True)
df.fillna({'X17':df['X17'].mean()},inplace=True)
df.fillna({'X15':df['X15'].mean()},inplace=True)
df.fillna({'X5':df['X5'].mean()},inplace=True)
df.fillna({'X9':df['X9'].mean()},inplace=True)
df.fillna({'X13':df['X13'].mean()},inplace=True)
df.fillna({'X23':df['X23'].mean()},inplace=True)
# 对于缺失值占比少的进行众数填充
df.fillna({'X21':df['X23'].mode()},inplace=True)
df.fillna({'X8':df['X8'].mode()},inplace=True)
df.fillna({'X11':df['X11'].mode()},inplace=True)

# # data.isnull()#缺失值判断：是缺失值返回True，否则范围False
# # data.isnull().sum()#缺失值计算：返回每列包含的缺失值的个数
# df.dropna()   #缺失值删除：直接删除含有缺失值的行
# # data.dropna(axis = 1)#缺失值删除列：直接删除含有缺失值的列
# # data.dropna(how = 'all')#缺失值删除行：只删除全是缺失值的行
# # data.dropna(thresh = n)#缺失值删除判断:保留至少有n个缺失值的行
# df.dropna(subset = ['X8']) #缺失值删除列：删除含有缺失值的特定的列

# 再次确认是否还有空值
print(df.isnull().sum())
# 查看样本分布
df['event生存状态'].replace(to_replace = '1', value = 1,inplace = True)
df['event生存状态'].replace(to_replace = '0', value = 0,inplace = True)
df['event生存状态'].head()
print(df['event生存状态'].head())

# # 绘制死亡率饼图
# event_value=df["event生存状态"].value_counts()
# labels=df["event生存状态"].value_counts().index
# plt.figure(figsize=(7,7))
# plt.pie(event_value,labels=labels,colors=["b","w"], explode=(0.1,0),autopct='%1.1f%%', shadow=True)
# plt.rcParams['font.sans-serif']=['Simhei'] # 添加此代码不让中文字符乱码
# plt.title("死亡患者占比高达37.0%")
# plt.show()
# 分析：死亡患者占比37.0%，存活患者占比63.0%，明显的“样本不均衡”。


# 提取特征
feature=df.iloc[:,1:26]
# 皮尔斯系数相关,查看变量间的两两相关性
corr_df = feature.apply(lambda x: pd.factorize(x)[0])
corr_df.head()
# 相关性矩阵
corr=corr_df.corr()
print(corr)
# 绘制热力图观察变量之间的相关性强弱
# plt.figure(figsize=(15,12))
# ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
#                  linewidths=0.2, cmap="RdYlGn",annot=True)
# plt.rcParams['font.sans-serif'] = ['Simhei']  # 添加此代码不让中文字符乱码
# plt.title("各个变量之间的相关性")
# plt.show()

# 独热编码
# 查看研究对象"event生存状态"与其他变量下的标签相关性。
# 独热编码，可以将分类变量下的标签转化成列
df_onehot = pd.get_dummies(df.iloc[:,1:27])
df_onehot.head()
# # 画图
# plt.figure(figsize=(15,6))
# df_onehot.corr()['event生存状态'].sort_values(ascending=False).plot(kind='bar')
# plt.rcParams['font.sans-serif'] = ['Simhei']  # 添加此代码不让中文字符乱码
# plt.title('生存状态与各个维度之间的关系')
# plt.show()


# 构建模型部分
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV # 划分数据集,网格搜索
from sklearn.tree import DecisionTreeClassifier # 分类树
from sklearn.neighbors import KNeighborsClassifier # k近邻算法   （sklearn.neighbors最邻近算法模块）
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.ensemble import RandomForestClassifier # 随机森林
from sklearn.metrics import precision_score, accuracy_score,recall_score, classification_report,f1_score
from sklearn.model_selection import learning_curve
import random

# 查看特征分布
df.groupby("event生存状态").size()
print(df.groupby("event生存状态").size())

# 模型预测
# 划分训练集测试集
X = df.drop(columns=["event生存状态"])
y = df["event生存状态"]

# 样本平衡处理：采用的是过采样SMOTE算法，其主要原理是通过插值的方式插入近邻的数据点。
print('样本个数:{},1占:{:.2%},0占:{:.2%}'.format(X.shape[0],y.sum()/X.shape[0],(X.shape[0]-y.sum())/X.shape[0]))
seed = 0
smote = SMOTE(random_state=seed,k_neighbors=10)
over_X,over_y = smote.fit_resample(X,y)
print('样本个数:{},1占:{:.2%},0占:{:.2%}'.format(over_X.shape[0],over_y.sum()/over_X.shape[0],(over_X.shape[0]-over_y.sum())/over_X.shape[0]))

Xtrain,Xtest,ytrain,ytest=train_test_split(over_X,over_y,test_size=0.3,random_state=10)

# 采用决策树、随机森林、K近邻、逻辑回归四个模型。观察模型的准确度、精确度、召回率、f1。
model = [DecisionTreeClassifier(random_state=120)
    , RandomForestClassifier(random_state=120)
    ,KNeighborsClassifier()
    , LogisticRegression(max_iter=1000)]
for clf in model:
    clf.fit(Xtrain, ytrain)  # 使用X作为训练数据，y作为标签目标数据进行数据拟合训练。
    y_pre = clf.predict(Xtest)  # 根据提供的数据去预测它的类别标签
    precision = precision_score(ytest, y_pre)
    accuracy = accuracy_score(ytest, y_pre)
    print(clf, '\n \n', classification_report(ytest, y_pre)
          , '\n \n Precision Score:', precision
          , '\n Accuracy Score::', accuracy
          , '\n\n')

DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0,
                       random_state=120, splitter='best')

RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=120,
                       verbose=0, warm_start=False)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

# 绘制学习曲线 观察模型的拟合情况。
figure,ax=plt.subplots(1,4,figsize=(30,4))
for i in range(4):
    train_sizes, train_scores, valid_scores=learning_curve(model[i],over_X,over_y, cv=5,random_state=10)
    train_std=train_scores.mean(axis=1)
    test_std=valid_scores.mean(axis=1)
    ax[i].plot(train_sizes,train_std,color='red',label='train_scores')
    ax[i].plot(train_sizes,test_std,color='blue',label='test_scores')
plt.rcParams['font.sans-serif'] = ['Simhei']  # 添加此代码不让中文字符乱码
plt.title('从左到右以此是决策树、随机森林、K近邻、逻辑回归学习曲线')
plt.legend()
plt.show()

# 决策树调优
# 利用网格搜索调整随机森林的参数
param_grid  = {'max_depth': range(1,10,2)
              }
rfc = DecisionTreeClassifier(random_state=120)
gridsearch = GridSearchCV(estimator =rfc, param_grid=param_grid,cv=5)
gridsearch.fit(Xtrain,ytrain)
print('best_params:',gridsearch.best_params_
      ,'\n \nbest_score: ',gridsearch.best_score_)
# best_params: {'max_depth': 1}
# best_score: 0.9849159663865545

# 经过参数的调整，计算F1的值。
rfc = DecisionTreeClassifier(random_state=120,max_depth=1)
rfc.fit(Xtrain,ytrain)
y_pre = rfc.predict(Xtest)
f1_score(ytest,y_pre)
print(f1_score(ytest,y_pre))
# precision = precision_score(ytest, y_pre)
# accuracy = accuracy_score(ytest, y_pre)
# print(precision)
# print(accuracy)
# 0.9847328244274809

rfc = DecisionTreeClassifier(random_state=120,max_depth=1)
train_sizes, train_scores, valid_scores=learning_curve(rfc,over_X,over_y, cv=5,random_state=10)
train_std = train_scores.mean(axis=1)
test_std = valid_scores.mean(axis=1)
plt.plot(train_sizes,train_std,color='red',label='train_scores')
plt.plot(train_sizes,test_std,color='blue',label='test_scores')
plt.legend()
plt.rcParams['font.sans-serif']=['Simhei'] # 添加此代码不让中文字符乱码
plt.title('决策树学习曲线')
plt.show()

# 逻辑回归需要调优的参数
penaltys = ['l1', 'l2']
Cs = [0.1, 1, 10, 100, 1000]
# 调优的参数集合，搜索网格为2x5，在网格上的交叉点进行搜索
tuned_parameters = dict(penalty=penaltys, C=Cs)
lr_penalty = LogisticRegression(solver='liblinear')
grid = GridSearchCV(lr_penalty, tuned_parameters, cv=3)
grid.fit(Xtrain, ytrain)
print('best_params:',grid.best_params_
      ,'\n \nbest_score: ',grid.best_score_)
# 运行结果：
# best_params: {'C': 0.1, 'penalty': 'l1'}
# best_score:  0.9765494137353433

# 经过参数的调整，计算F1的值。
rfc = LogisticRegression(random_state=120, C=1, penalty='l1', solver='liblinear')
rfc.fit(Xtrain,ytrain)
y_pre = rfc.predict(Xtest)
f1_score(ytest,y_pre)
# precision = precision_score(ytest, y_pre)
# accuracy = accuracy_score(ytest, y_pre)
# print(precision)
# print(accuracy)
print(f1_score(ytest,y_pre))
# 0.9883268482490272

rfc = LogisticRegression(random_state=120, C=1, penalty='l1')
train_sizes, train_scores, valid_scores=learning_curve(rfc,over_X,over_y, cv=5,random_state=10)
train_std = train_scores.mean(axis=1)
test_std = valid_scores.mean(axis=1)
plt.plot(train_sizes,train_std,color='red',label='train_scores')
plt.plot(train_sizes,test_std,color='blue',label='test_scores')
plt.legend()
plt.rcParams['font.sans-serif']=['Simhei'] # 添加此代码不让中文字符乱码
plt.title('逻辑回归学习曲线')
plt.show()

# KNN网格搜索
n_neighbors = [3,5,7,9]
p = [1,2]
tuned_parameters = dict(n_neighbors=n_neighbors, p=p )
lr_n_neighbors = KNeighborsClassifier()
grid = GridSearchCV(lr_n_neighbors, tuned_parameters, cv=3)
grid.fit(Xtrain, ytrain)
print('best_params:',grid.best_params_
      ,'\n \nbest_score: ',grid.best_score_)
# 运行结果：
# best_params: {'n_neighbors': 5, 'p': 1}
# best_score: 0.6247906197654941

# 经过参数的调整，计算F1的值。
rfc = KNeighborsClassifier(n_neighbors=5, p=1)
rfc.fit(Xtrain,ytrain)
y_pre = rfc.predict(Xtest)
f1_score(ytest,y_pre)
# precision = precision_score(ytest, y_pre)
# accuracy = accuracy_score(ytest, y_pre)
# print(precision)
# print(accuracy)
print(f1_score(ytest,y_pre))
# 0.6561264822134388

rfc = KNeighborsClassifier(n_neighbors=5, p=1)
train_sizes, train_scores, valid_scores=learning_curve(rfc,over_X,over_y, cv=5,random_state=10)
train_std = train_scores.mean(axis=1)
test_std = valid_scores.mean(axis=1)
plt.plot(train_sizes,train_std,color='red',label='train_scores')
plt.plot(train_sizes,test_std,color='blue',label='test_scores')
plt.legend()
plt.rcParams['font.sans-serif']=['Simhei'] # 添加此代码不让中文字符乱码
plt.title('K近邻学习曲线')
plt.show()


# 利用网格搜索调整随机森林的参数
param_grid  = {'n_estimators' : [400,1200],
                'max_depth': range(1,10,2),
              }
rfc = RandomForestClassifier(random_state=120)
gridsearch = GridSearchCV(estimator =rfc, param_grid=param_grid,cv=5)
gridsearch.fit(Xtrain,ytrain)
print('best_params:',gridsearch.best_params_
      ,'\n \nbest_score: ',gridsearch.best_score_)
# best_params: {'max_depth': 5, 'n_estimators': 1200}
# best_score: 0.9882492997198881

# 经过参数的调整，计算F1的值。
rfc = RandomForestClassifier(random_state=120,max_depth=5, n_estimators=1200)
rfc.fit(Xtrain,ytrain)
y_pre = rfc.predict(Xtest)
f1_score(ytest,y_pre)
precision = precision_score(ytest, y_pre)
# accuracy = accuracy_score(ytest, y_pre)
# print(precision)
# print(accuracy)
print(f1_score(ytest,y_pre))
# 0.9961389961389961

# rfc = RandomForestClassifier(random_state=120,max_depth=5, n_estimators=1200)
# train_sizes, train_scores, valid_scores=learning_curve(rfc,over_X,over_y, cv=5,random_state=10)
# train_std = train_scores.mean(axis=1)
# test_std = valid_scores.mean(axis=1)
# plt.plot(train_sizes,train_std,color='red',label='train_scores')
# plt.plot(train_sizes,test_std,color='blue',label='test_scores')
# plt.legend()
# plt.rcParams['font.sans-serif']=['Simhei'] # 添加此代码不让中文字符乱码
# plt.title('随机森林学习曲线')
# plt.show()


# # 特征的重要性
# # 查看特征的重要性并排序
# fea_import=pd.DataFrame(rfc.feature_importances_)
# fea_import['feature']=list(Xtrain)
# fea_import.sort_values(0,ascending=False,inplace=True)
# fea_import.reset_index(drop=True,inplace=True)
# fea_import
# print(fea_import)
# # 将其可视化
# figuer,ax=plt.subplots(figsize=(12,8))
# g=sns.barplot(0,'feature',data=fea_import)
# g.set_xlabel('Weight')
# g.set_title('Random Forest')
# plt.rcParams['font.sans-serif']=['Simhei'] # 添加此代码不让中文字符乱码
# plt.show()




















