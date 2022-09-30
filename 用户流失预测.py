import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'D:\pythonProject\机械学习—电信用户流失预测\WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.columns=['用户ID','性别','老年人' ,'是否有配偶' ,'是否经济独立' ,'客户的入网时间','是否开通电话服务业务' ,'是否开通了多线业务'
,'是否开通互联网服务' ,'是否开通网络安全服务','是否开通在线备份业务','是否开通了设备保护业务','是否开通了技术支持服务','是否开通网络电视'
,'是否开通网络电影','签订合同年限' ,'是否开通电子账单','付款方式','月费用','总费用','该用户是否流失']
data.head()

# plt.figure(figsize=(15,8))
# data.corr()['流失'].sort_values(ascending = False).plot(kind='bar')
# plt.show()

# 强制转换为数字，不可转换的变为NaN
data['总费用'] = pd.to_numeric(data['总费用'],errors='coerce')

test=data.loc[:,'总费用'].value_counts().sort_index()
print(test.sum())
#运行结果：7032
print(data.客户的入网时间[data['总费用'].isnull().values==True])
#运行结果：11
# data.loc[:,'总费用'].replace(to_replace=np.nan,value=data.loc[:,'月费用'],inplace=True)#查看是否替换成功
# print(data[data['该用户是否流失']==0][['该用户是否流失','月费用','总费用']])

data.loc[:,'该用户是否流失'].replace(to_replace=0,value=1,inplace=True)
print(pd.isnull(data['总费用']).sum())
print(data['总费用'].dtypes)
# 填充缺失值
data.fillna({"总费用":data["总费用"].median()},inplace=True)
data.isnull().sum(axis=0)
print(data.isnull().sum(axis=0))


# 根据一般经验，将用户特征划分为用户属性、服务属性、合同属性，并从这三个维度进行分析。
# 查看流失用户数量和占比。
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.pie(data['该用户是否流失'].value_counts(),labels=['未流失','流失'],autopct='%.2f%%',explode=(0.1,0))
plt.title('用户流失占比')
plt.show()
# 查看数量上的对比
churnDf=data['该用户是否流失'].value_counts().to_frame()
x=churnDf.index
y=churnDf['该用户是否流失']
plt.bar(x,y,width = 0.5,color = 'c')
# 用来正常显示中文标签（需要安装字库）
plt.title('该用户是否流失(Yes/No) Num')
plt.show()
# 属于不平衡数据集，流失用户占比达26.54%。

# 用户属性分析
def plt_bar(featrue):
    df=(data.groupby(data[featrue])['该用户是否流失'].value_counts()/len(data)).unstack()
    df.plot(kind='bar')
    plt.title('{}流失分析'.format(featrue),fontsize=15)
    plt.xticks(rotation=0)
    plt.rcParams.update({'font.size': 10})
plt_bar('性别')
plt_bar('老年人')
plt.show()
# 小结： 用户流失基本与性别无关。 年老用户的流失占比显著高于年轻用户。


def plt_kde(feature):
    x=data[data['该用户是否流失']=='Yes'][feature]
    y=data[data['该用户是否流失']=='No'][feature]
#参数如下：
# sns.kdeplot(data,data2=None,shade=False,vertical=False,kernel='gau',bw='scott',gridsize=100,cut=3,clip=None,
#             legend=True,cumulative=False,shade_lowest=True,cbar=False, cbar_ax=None, cbar_kws=None, ax=None, *kwargs)
# 核密度估计是概率论上用来估计未知的密度函数，属于非参数检验，通过核密度估计图可以比较直观的看出样本数据本身的分布特征
    sns.kdeplot(x,shade=True,label='已流失客户')
    sns.kdeplot(y,shade=True,label='未流失客户')
    plt.title('{}分析'.format(feature),fontsize=15)
plt_kde('客户的入网时间')
plt_bar('是否有配偶')
plt_bar('是否经济独立')
plt.show()
# 小结：
# 用户的入网时间越短，流失越高，符合一般规律。
# 用户入网时间在0-20天内流失风险最大。
# 入网时间达到三个月，流失率小于在网率，证明用户心理的稳定期一般是3个月。
# 小结：
# 没有配偶的用户流失率高于有配偶。
# 经济未独立的用户流失率高于经济独立。

# 3.服务属性分析
def plt_hbar(feature1, feature2):
    figure, ax = plt.subplots(2, 1, figsize=(12, 12))
    gp_dep = (data.groupby(feature1)['该用户是否流失'].value_counts() / len(data)).to_frame()
    gp_dep.rename(columns={"该用户是否流失": '占比'}, inplace=True)
    gp_dep.reset_index(inplace=True)
    sns.barplot(x='占比', hue='该用户是否流失', y=feature1, data=gp_dep, ax=ax[0])

    gp = (data.groupby(feature2)['该用户是否流失'].value_counts() / len(data)).to_frame()
    gp.rename(columns={"该用户是否流失": '占比'}, inplace=True)
    gp.reset_index(inplace=True)
    sns.barplot(x='占比', hue='该用户是否流失', y=feature2, data=gp, ax=ax[1])

plt_hbar("是否开通了多线业务", "是否开通互联网服务")
plt.show()

df1=pd.melt(data[data['是否开通互联网服务']!='No'][['是否开通电话服务业务','是否开通了多线业务','是否开通网络安全服务'
 ,'是否开通在线备份业务','是否开通了设备保护业务','是否开通了技术支持服务'
,'是否开通网络电视','是否开通网络电影']],value_name='有服务')
plt.figure(figsize=(20, 8))
sns.countplot(data=df1,x='variable',hue='有服务')
plt.show()

df2=pd.melt(data[(data['是否开通互联网服务']!='No')& (data['该用户是否流失']=='Yes')][['是否开通电话服务业务','是否开通了多线业务','是否开通网络安全服务'
 ,'是否开通在线备份业务','是否开通了设备保护业务','是否开通了技术支持服务'
,'是否开通网络电视','是否开通网络电影']])
figure,ax=plt.subplots(figsize=(16,8))
sns.countplot(x='variable',hue='value',data=df2)
plt.show()
# 小结：
# 电话服务对整体的流失影响较小。
# 单光纤用户的流失占比较高。
# 光纤用户绑定了安全、备份、保护、技术支持服务的流失率较低
# 光纤用户附加网络电视、电影服务的流失率占比较高。

# 4.合同属性分析
plt_bar('签订合同年限')
plt_bar('是否开通电子账单')
plt.show()

data['是否流失']=data['该用户是否流失'].replace('No',0).replace('Yes',1)
g = sns.FacetGrid(data, col="是否开通电子账单", height=6, aspect=.9)
ax = g.map(sns.barplot, "签订合同年限","是否流失",palette = "Blues_d"
           , order= ['Month-to-month', 'One year', 'Two year'])
plt.show()

figure,ax=plt.subplots(figsize=(12,6))
gp_dep=(data.groupby('付款方式')['该用户是否流失'].value_counts()/len(data)).to_frame()
gp_dep.rename(columns={"该用户是否流失":'占比'} , inplace=True)
gp_dep.reset_index(inplace=True)
sns.barplot(x='占比', hue='该用户是否流失',y='付款方式', data=gp_dep)
plt.show()

plt_kde('月费用')
plt.show()
plt_kde('总费用')
plt.show()
# 签订合同年限的用户流失率：按月签订>按一年签订>按两年签订，证明长期合同能保留用户。
# 开通电子账单的用户流失率较高，原因是采用电子支票的用户按月签订合同的占比较高，用户黏性低，易流失。
# 采用电子支票的用户流失率最高，推测是该方式的使用体验一般。
# 月消费金额大约在70-110用户流失率较高。
# 长期来看，用户总消费越高，流失率越低，符合一般经验。
# 通过以上分析，可以得到较高流失率的人群特征，具有这些特征的人群需要对其经营，增加用户黏性，延长生命周期价值。

# 八、用户流失预测
# 1.数据预处理，导包
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split,GridSearchCV #划分数据集,网格搜索
from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn.neighbors import KNeighborsClassifier #k近邻算法        （sklearn.neighbors最邻近算法模块）
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.metrics import precision_score, accuracy_score,recall_score, classification_report,f1_score
from sklearn.model_selection import learning_curve
import random

# 用户ID列没用，将其删除。
data.drop('用户ID',inplace=True,axis=1)
data.info()


data['是否开通了多线业务'].replace('No phone service','No',inplace=True)
col=['是否开通网络安全服务','是否开通在线备份业务','是否开通了设备保护业务',
     '是否开通了技术支持服务','是否开通网络电视','是否开通网络电影']
for i in col:
    data[i].replace('No internet service','No',inplace=True)
# 观察数据类型，发现除了“客户的入网时间”、“月费用”、“总费用”是连续型特征，
# 其余基本是离散型特征，对于离散型特征，特征之间没有大小关系，采用one-hot编码，特征之间有大小关系，采用数值映射。
data_onehot = pd.get_dummies(data.iloc[:,1:21])
data_onehot.head()
# 绘图查看 用户流失(‘Churn’)与各个维度之间的关系
# plt.figure(figsize=(15,6))
# data_onehot.corr()['流失'].sort_values(ascending=False).plot(kind='bar')
# plt.title('流失率和变量之间的相关性')
# plt.show()

clos=data[data.columns[data.dtypes==object]].copy()
for i in clos.columns:
    if clos[i].nunique()==2:
        clos[i]=pd.factorize(clos[i])[0]
        print(clos[i])  # pd.factorize功能转换为0,1或2等
# factorize()对每一个类别映射一个ID，这种映射最后只生成一个特征，
# get_dummies()映射后生成多个特征,即将签订合同年限分为3列。
    else:
        clos=pd.get_dummies(clos,columns=[i])
clos['老年人']=data['老年人']
clos['月费用']=data['月费用']
clos['总费用']=data['总费用']
clos['客户的入网时间']=data['客户的入网时间']

# 划分特征和标签。
X=clos.iloc[:,clos.columns!='该用户是否流失']
y=clos.iloc[:,clos.columns=='该用户是否流失'].values.ravel()

# 样本平衡处理：采用的是过采样SMOTE算法，其主要原理是通过插值的方式插入近邻的数据点。
print('样本个数:{},1占:{:.2%},0占:{:.2%}'.format(X.shape[0],y.sum()/X.shape[0],(X.shape[0]-y.sum())/X.shape[0]))
seed=0
smote=SMOTE(random_state=seed,k_neighbors=10)
over_X,over_y=smote.fit_resample(X,y)
print('样本个数:{},1占:{:.2%},0占:{:.2%}'.format(over_X.shape[0],over_y.sum()/over_X.shape[0],(over_X.shape[0]-over_y.sum())/over_X.shape[0]))
# 运行结果：
# 样本个数:7043,1占:26.54%,0占:73.46%
# 样本个数:10348,1占:50.00%,0占:50.00%

# 模型预测
# 划分训练集测试集。
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
# 运行结果：
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
plt.legend()
plt.show()

# 需要调优的参数
# 请尝试将L1正则和L2正则分开，并配合合适的优化求解算法（solver）
# tuned_parameters={'penalth':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}
# 参数的搜索范围
penaltys = ['l1', 'l2']
Cs = [0.1, 1, 10, 100, 1000]
# 调优的参数集合，搜索网格为2x5，在网格上的交叉点进行搜索
tuned_parameters = dict(penalty=penaltys, C=Cs)
lr_penalty = LogisticRegression(solver='liblinear')
grid = GridSearchCV(lr_penalty, tuned_parameters, cv=3)
grid.fit(Xtrain, ytrain)
print('best_params:',grid.best_params_
      ,'\n \nbest_score: ',grid.best_score_)
y_pre=grid.predict(Xtest)
f1_score(ytest,y_pre)
print(f1_score(ytest,y_pre))


# KNN网格搜索
n_neighbors=[3,5,7,9]
p=[1,2]
tuned_parameters = dict(n_neighbors=n_neighbors, p=p )
lr_n_neighbors=KNeighborsClassifier()
grid = GridSearchCV(lr_n_neighbors, tuned_parameters, cv=3)
grid.fit(Xtrain, ytrain)
print('best_params:',grid.best_params_
      ,'\n \nbest_score: ',grid.best_score_)
y_pre=grid.predict(Xtest)
f1_score(ytest,y_pre)
print(f1_score(ytest,y_pre))




# 从结果来看，随机森林模型预测的准确度、精度、以及f1分数较高，
# 但是却有严重的过拟合。接下来利用网格搜索调整随机森林的参数，降低过拟合。
param_grid  = {
                'n_estimators' : [400,1200],
#                'min_samples_split': [2,5,10,15],
#                'min_samples_leaf': [1,2,5,10],
                'max_depth': range(1,10,2),
#                 'max_features' : ('log2', 'sqrt'),
              }
rfc=RandomForestClassifier(random_state=120)
gridsearch = GridSearchCV(estimator =rfc, param_grid=param_grid,cv=5)
gridsearch.fit(Xtrain,ytrain)
print('best_params:',gridsearch.best_params_
      ,'\n \nbest_score: ',gridsearch.best_score_)
# #运行结果：
# best_params: {'max_depth': 9, 'n_estimators': 500}
# best_score:  0.8408114378748538

rfc=RandomForestClassifier(random_state=120,max_depth=9, n_estimators= 500)
train_sizes, train_scores, valid_scores=learning_curve(rfc,over_X,over_y, cv=5,random_state=10)
train_std=train_scores.mean(axis=1)
test_std=valid_scores.mean(axis=1)
plt.plot(train_sizes,train_std,color='red',label='train_scores')
plt.plot(train_sizes,test_std,color='blue',label='test_scores')
plt.legend()
plt.show()

# 好了，经过参数的调整，过拟合的情况已经明显降低了，计算F1的值。
rfc=RandomForestClassifier(random_state=120,max_depth=9, n_estimators= 500)
rfc.fit(Xtrain,ytrain)
y_pre=rfc.predict(Xtest)
f1_score(ytest,y_pre)
print(f1_score(ytest,y_pre))
# 运行结果：0.8451776649746194


# 3.特征的重要性
# 查看特征的重要性并排序。
fea_import=pd.DataFrame(rfc.feature_importances_)
fea_import['feature']=list(Xtrain)
fea_import.sort_values(0,ascending=False,inplace=True)
fea_import.reset_index(drop=True,inplace=True)
fea_import
print(fea_import)
# 将其可视化。
figuer,ax=plt.subplots(figsize=(12,8))
g=sns.barplot(0,'feature',data=fea_import)
g.set_xlabel('Weight')
g.set_title('Random Forest')
plt.show()







# # 画出模型的学习曲线，观察模型的拟合情况。
# figure,ax=plt.subplots(1,4,figsize=(30,4))
# for i in range(4):
#     train_sizes, train_scores, valid_scores=learning_curve(model[i],over_X,over_y, cv=5,random_state=10)
#     train_std=train_scores.mean(axis=1)
#     test_std=valid_scores.mean(axis=1)
#     ax[i].plot(train_sizes,train_std,color='red',label='train_scores')
#     ax[i].plot(train_sizes,test_std,color='blue',label='test_scores')
# plt.legend()
#
#
#
# # 从结果来看，随机森林模型预测的准确度、精度、以及f1分数较高，
# # 但是却有严重的过拟合。接下来利用网格搜索调整随机森林的参数，降低过拟合
# param_grid  = {
#                 'n_estimators' : [500,1200],
# #                'min_samples_split': [2,5,10,15],
# #                'min_samples_leaf': [1,2,5,10],
#                 'max_depth': range(1,10,2),
# #                 'max_features' : ('log2', 'sqrt'),
#               }
# rfc=RandomForestClassifier(random_state=120)
# gridsearch = GridSearchCV(estimator =rfc, param_grid=param_grid,cv=5)
# gridsearch.fit(Xtrain,ytrain)
# print('best_params:',gridsearch.best_params_
#       ,'\n \nbest_score: ',gridsearch.best_score_)
# #运行结果：
# # best_params: {'max_depth': 9, 'n_estimators': 500}
# # best_score:  0.8408114378748538
#
# # 好了，经过参数的调整，过拟合的情况已经明显降低了，计算F1的值。
# rfc=RandomForestClassifier(random_state=120,max_depth=9, n_estimators= 500)
# rfc.fit(Xtrain,ytrain)
# y_pre=rfc.predict(Xtest)
# f1_score(ytest,y_pre)
# #运行结果：0.8451776649746194
#
# # 3.特征的重要性
# # 查看特征的重要性并排序。
# fea_import=pd.DataFrame(rfc.feature_importances_)
# fea_import['feature']=list(Xtrain)
# fea_import.sort_values(0,ascending=False,inplace=True)
# fea_import.reset_index(drop=True,inplace=True)
# fea_import
# figuer,ax=plt.subplots(figsize=(12,8))
# g=sns.barplot(0,'feature',data=fea_import)
# g.set_xlabel('Weight')
# g.set_title('Random Forest')










