import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

#Definition functions
def dist(vec1, vec2):
    return np.sum(np.square(np.linalg.norm(vec1 - vec2)))

def SSE(cluster):
    sigma1 = 0
    size1 = cluster.shape[0]
    center11 = np.sum(cluster, axis=0) / cluster.shape[0]
    for i in range(size1):
                sigma1 += dist(cluster[i],center11)
    return sigma1

def MSE(cluster):
    sigma1 = 0
    size1 = cluster.shape[0]
    center11 = np.sum(cluster, axis=0) / cluster.shape[0]
    for i in range(size1):
                sigma1 += dist(cluster[i],center11)
    return sigma1/size1

#K-P
    
#Read data
dff2 = pd.read_excel('E:/adult.xlsx')

#Features of interest (FOI)
dff2[['age']].describe()

dff2.loc[dff2['age']<=40,'class2']=1000
dff2.loc[dff2['age']>=41,'class3']=2000
dff2.loc[dff2['class']=='<=50K','class0']=1000
dff2.loc[dff2['class']=='>50K','class1']=2000

dff2.loc[dff2['class2']==dff2['class0'],'lb']=0
dff2.loc[dff2['class2']<dff2['class1'],'lb']=1
dff2.loc[dff2['class3']>dff2['class0'],'lb']=2
dff2.loc[dff2['class3']==dff2['class1'],'lb']=3
dff3=dff2.drop(columns=['class','class0','class1','class2','class3'])

dff=dff3.iloc[:,0:6]
dff = dff.astype(int)
print(np.sum(dff.isna()))
print(dff.dtypes)
print(dff.shape)
dfff=dff

#K-P (1-Normalize K-means)
start = timeit.default_timer()
#Standard deviation check for the decision to normalize
print (dfff.agg('std'))
#Normalize
dffff=dfff
standardize=StandardScaler()
standardize.fit(dffff)
dffff=standardize.fit_transform(dffff)
dffff=pd.DataFrame(dffff)
dfff=dffff
#Run K-means
kmeans = KMeans(n_clusters=5).fit(dfff)
labels=kmeans.labels_

#Sil
SC=[]
n_clusters=5
silhouette_score=metrics.silhouette_score(dfff, labels, metric='euclidean')
sample_silhouette_values = metrics.silhouette_samples(dfff, labels)
means_lst = []
for label in range(n_clusters):
    means_lst.append(sample_silhouette_values[labels == label].mean())
means_lst

#...K-means
dfff['clusters']=kmeans.labels_
print(dfff.groupby('clusters').agg('mean'))
print (dfff.agg('std'))
c0=dfff[dfff['clusters']==0]
c0=np.array(c0.iloc[:,0:6])
c1=dfff[dfff['clusters']==1]
c1=np.array(c1.iloc[:,0:6])
c2=dfff[dfff['clusters']==2]
c2=np.array(c2.iloc[:,0:6])
c3=dfff[dfff['clusters']==3]
c3=np.array(c3.iloc[:,0:6])
c4=dfff[dfff['clusters']==4]
c4=np.array(c4.iloc[:,0:6])

#MSE for each cluster
print('MSE 0',MSE(c0))
print('MSE 1',MSE(c1))
print('MSE 2',MSE(c2))
print('MSE 3',MSE(c3))
print('MSE 4',MSE(c4))
#SSE/N
print('WSSE kol fml',(SSE(c0)+SSE(c1)+SSE(c2)+SSE(c3)+SSE(c4))/dff.shape[0])
#Average MSE
print('AVMSE kol fml',(MSE(c0)+MSE(c1)+MSE(c2)+MSE(c3)+MSE(c4))/5)
#Sil
print('silhouette_score',silhouette_score)
#Sil clusters
print('means_lst',means_lst)
#Interpretability
lb=dff3.iloc[:,6]
lb0=lb.index[lb==0]
lb1=lb.index[lb==1]
lb2=lb.index[lb==2]
lb3=lb.index[lb==3]
cl=dfff['clusters']
cl0=cl.index[cl==0]
cl1=cl.index[cl==1]
cl2=cl.index[cl==2]
cl3=cl.index[cl==3]
cl4=cl.index[cl==4]
lbmap = pd.DataFrame()
lbmap['dff_index'] = dff.index.values
lbmap['cl'] = kmeans.labels_
lbmap['lb'] = lb
flb0cl0=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==0)])))/(len(cl0))
flb1cl0=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==0)])))/(len(cl0))
flb2cl0=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==0)])))/(len(cl0))
flb3cl0=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==0)])))/(len(cl0))

flb0cl1=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==1)])))/(len(cl1))
flb1cl1=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==1)])))/(len(cl1))
flb2cl1=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==1)])))/(len(cl1))
flb3cl1=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==1)])))/(len(cl1))

flb0cl2=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==2)])))/(len(cl2))
flb1cl2=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==2)])))/(len(cl2))
flb2cl2=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==2)])))/(len(cl2))
flb3cl2=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==2)])))/(len(cl2))

flb0cl3=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==3)])))/(len(cl3))
flb1cl3=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==3)])))/(len(cl3))
flb2cl3=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==3)])))/(len(cl3))
flb3cl3=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==3)])))/(len(cl3))

flb0cl4=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==4)])))/(len(cl4))
flb1cl4=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==4)])))/(len(cl4))
flb2cl4=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==4)])))/(len(cl4))
flb3cl4=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==4)])))/(len(cl4))
print('cl0')
print('flb0cl0',flb0cl0)
print('flb1cl0',flb1cl0)
print('flb2cl0',flb2cl0)
print('flb3cl0',flb3cl0)
print('cl1')
print('flb0cl1',flb0cl1)
print('flb1cl1',flb1cl1)
print('flb2cl1',flb2cl1)
print('flb3cl1',flb3cl1)
print('cl2')
print('flb0cl2',flb0cl2)
print('flb1cl2',flb1cl2)
print('flb2cl2',flb2cl2)
print('flb3cl2',flb3cl2)
print('cl3')
print('flb0cl3',flb0cl3)
print('flb1cl3',flb1cl3)
print('flb2cl3',flb2cl3)
print('flb3cl3',flb3cl3)
print('cl4')
print('flb0cl4',flb0cl4)
print('flb1cl4',flb1cl4)
print('flb2cl4',flb2cl4)
print('flb3cl4',flb3cl4)

#K-P (2-Check MSE and creat sub-dataset 1)
df=dfff[dfff['clusters']!=0]#,#4
df=df[df['clusters']!=1]#,#4
df=df[df['clusters']!=4]#,#4

#...Sil...
SC.append(means_lst[0])#,#4
SC.append(means_lst[1])#,#4
SC.append(means_lst[4])#,#4

#K-P (3-PCA)
df=df.iloc[:,0:6]
df2f=df
X1=df
X=df
p=PCA()
p.fit(X)
W=p.components_.T
y=p.fit_transform(X)
yhat=X.dot(W)
yhat.columns
#K-P (4-EV)
w=pd.DataFrame(W[:,:6],index=df.columns[:])
EV=pd.DataFrame(p.explained_variance_ratio_,index=np.arange(6)+1,columns=['Explained Variability'])
Sev=np.cumsum(p.explained_variance_ratio_)
print(w)
print(EV)
print(Sev)

#K-P (5-Check EV and reduce dimension)
dforg=yhat.iloc[:,:5]##

#K-P (6-Run K-means)
kmeans1 = KMeans(n_clusters=2).fit(dforg)#4
labels1=kmeans1.labels_

#...Sil...
n_clusters=2#4
sample_silhouette_values = metrics.silhouette_samples(dforg, labels1)
means_lst = []
for label in range(n_clusters):
    means_lst.append(sample_silhouette_values[labels1 == label].mean())
means_lst
SCF=means_lst+SC

#...K-means
dforg=pd.DataFrame(dforg)
dforg['clusters']=kmeans1.labels_
print(dforg.groupby('clusters').agg('mean'))
o0=dforg[dforg['clusters']==0]
o0=np.array(o0.iloc[:,:5])##
o1=dforg[dforg['clusters']==1]
o1=np.array(o1.iloc[:,:5])##
#MSE
print('MSE n0',MSE(o0))
print('MSE n1',MSE(o1))
#SSE/N
print('WSSE kol fml',(SSE(o0)+SSE(o1)+SSE(c0)+SSE(c1)+SSE(c4))/dff.shape[0])
#Average MSE
print('AVMSE kol fml',(MSE(o0)+MSE(o1))/2)
#Sil Finall
print('SCFmean',np.mean(SCF))
#interoperability
cll=dforg['clusters']
cll0=cll.index[cll==0]
cll1=cll.index[cll==1]
dflbb=pd.DataFrame(dff3,index=df.index,columns=dff3.columns)
lbb=dflbb.iloc[:,6]
lbb0=lbb.index[lbb==0]
lbb1=lbb.index[lbb==1]
lbb2=lbb.index[lbb==2]
lbb3=lbb.index[lbb==3]
lbbmap = pd.DataFrame()
lbbmap['df_index'] = df.index.values
lbbmap['cll'] = kmeans1.labels_
lbbmap['lbb'] = lb
flbb0cll0=(len((lbbmap.loc[(lbbmap['lbb']==0)&(lbbmap['cll']==0)])))/(len(cll0))
flbb1cll0=(len((lbbmap.loc[(lbbmap['lbb']==1)&(lbbmap['cll']==0)])))/(len(cll0))
flbb2cll0=(len((lbbmap.loc[(lbbmap['lbb']==2)&(lbbmap['cll']==0)])))/(len(cll0))
flbb3cll0=(len((lbbmap.loc[(lbbmap['lbb']==3)&(lbbmap['cll']==0)])))/(len(cll0))
flbb0cll1=(len((lbbmap.loc[(lbbmap['lbb']==0)&(lbbmap['cll']==1)])))/(len(cll1))
flbb1cll1=(len((lbbmap.loc[(lbbmap['lbb']==1)&(lbbmap['cll']==1)])))/(len(cll1))
flbb2cll1=(len((lbbmap.loc[(lbbmap['lbb']==2)&(lbbmap['cll']==1)])))/(len(cll1))
flbb3cll1=(len((lbbmap.loc[(lbbmap['lbb']==3)&(lbbmap['cll']==1)])))/(len(cll1))

print('cl0')
print('flbb0cll0',flbb0cll0)
print('flbb1cll0',flbb1cll0)
print('flbb2cll0',flbb2cll0)
print('flbb3cll0',flbb3cll0)
print('cl1')
print('flbb0cll1',flbb0cll1)
print('flbb1cll1',flbb1cll1)
print('flbb2cll1',flbb2cll1)
print('flbb3cll1',flbb3cll1)

stop = timeit.default_timer()
print('Time: ', stop - start)


#P-K

#Read data
dff2 = pd.read_excel('E:/adult.xlsx')

#Features of interest (FOI)
dff2[['age']].describe()

dff2.loc[dff2['age']<=40,'class2']=1000
dff2.loc[dff2['age']>=41,'class3']=2000
dff2.loc[dff2['class']=='<=50K','class0']=1000
dff2.loc[dff2['class']=='>50K','class1']=2000

dff2.loc[dff2['class2']==dff2['class0'],'lb']=0
dff2.loc[dff2['class2']<dff2['class1'],'lb']=1
dff2.loc[dff2['class3']>dff2['class0'],'lb']=2
dff2.loc[dff2['class3']==dff2['class1'],'lb']=3
dff3=dff2.drop(columns=['class','class0','class1','class2','class3'])
dff=dff3.iloc[:,0:6]
dff = dff.astype(int)
print(np.sum(dff.isna()))
print(dff.dtypes)
print(dff.shape)
dfff=dff

start = timeit.default_timer()
#Standard deviation check for the decision to normalize
print (dfff.agg('std'))
#Normalize
dffff=dfff
standardize=StandardScaler()
standardize.fit(dffff)
dffff=standardize.fit_transform(dffff)
dffff=pd.DataFrame(dffff)
dfff=dffff
#P-K (1-PCA)
X=dfff
p=PCA()
p.fit(X)
W=p.components_.T
y=p.fit_transform(X)
yhat=X.dot(W)
yhat.columns
#P-K (2-EV)
w=pd.DataFrame(W[:,:6],index=dfff.columns[:])
EV=pd.DataFrame(p.explained_variance_ratio_,index=np.arange(6)+1,columns=['Explained Variability'])
Sev=np.cumsum(p.explained_variance_ratio_)
print(w)
print(EV)
print(Sev)
#P-K (3-Check EV and reduce dimension)
dfff=yhat.iloc[:,:5]##
dfff=pd.DataFrame(dfff)
dfff.columns
#P-K (4-Run K-means)
kmeans = KMeans(n_clusters=5).fit(dfff)
labels=kmeans.labels_

#Sil
SC=[]
n_clusters=5
sample_silhouette_values = metrics.silhouette_samples(dfff, labels)
means_lst = []
for label in range(n_clusters):
    means_lst.append(sample_silhouette_values[labels == label].mean())
means_lst

#...K-means
dfff['clusters']=kmeans.labels_
print(dfff.groupby('clusters').agg('mean'))
c0=dfff[dfff['clusters']==0]
c0=np.array(c0.iloc[:,0:5])##
c1=dfff[dfff['clusters']==1]
c1=np.array(c1.iloc[:,0:5])##
c2=dfff[dfff['clusters']==2]
c2=np.array(c2.iloc[:,0:5])##
c3=dfff[dfff['clusters']==3]
c3=np.array(c3.iloc[:,0:5])##
c4=dfff[dfff['clusters']==4]
c4=np.array(c4.iloc[:,0:5])##
#MSE for each cluster
print('MSE 0',MSE(c0))
print('MSE 1',MSE(c1))
print('MSE 2',MSE(c2))
print('MSE 3',MSE(c3))
print('MSE 4',MSE(c4))
#SSE/N
print('WSSE kol fml',(SSE(c0)+SSE(c1)+SSE(c2)+SSE(c3)+SSE(c4))/dff.shape[0])
#Average MSE
print('AVMSE kol fml',(MSE(c0)+MSE(c1)+MSE(c2)+MSE(c3)+MSE(c4))/5)
#Sil clusters
print('means_lst',means_lst)
#Interpretability
lb=dff3.iloc[:,6]
lb0=lb.index[lb==0]
lb1=lb.index[lb==1]
lb2=lb.index[lb==2]
lb3=lb.index[lb==3]
cl=dfff['clusters']
cl0=cl.index[cl==0]
cl1=cl.index[cl==1]
cl2=cl.index[cl==2]
cl3=cl.index[cl==3]
cl4=cl.index[cl==4]
lbmap = pd.DataFrame()
lbmap['dff_index'] = dff.index.values
lbmap['cl'] = kmeans.labels_
lbmap['lb'] = lb
flb0cl0=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==0)])))/(len(cl0))
flb1cl0=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==0)])))/(len(cl0))
flb2cl0=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==0)])))/(len(cl0))
flb3cl0=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==0)])))/(len(cl0))

flb0cl1=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==1)])))/(len(cl1))
flb1cl1=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==1)])))/(len(cl1))
flb2cl1=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==1)])))/(len(cl1))
flb3cl1=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==1)])))/(len(cl1))

flb0cl2=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==2)])))/(len(cl2))
flb1cl2=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==2)])))/(len(cl2))
flb2cl2=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==2)])))/(len(cl2))
flb3cl2=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==2)])))/(len(cl2))

flb0cl3=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==3)])))/(len(cl3))
flb1cl3=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==3)])))/(len(cl3))
flb2cl3=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==3)])))/(len(cl3))
flb3cl3=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==3)])))/(len(cl3))

flb0cl4=(len((lbmap.loc[(lbmap['lb']==0)&(lbmap['cl']==4)])))/(len(cl4))
flb1cl4=(len((lbmap.loc[(lbmap['lb']==1)&(lbmap['cl']==4)])))/(len(cl4))
flb2cl4=(len((lbmap.loc[(lbmap['lb']==2)&(lbmap['cl']==4)])))/(len(cl4))
flb3cl4=(len((lbmap.loc[(lbmap['lb']==3)&(lbmap['cl']==4)])))/(len(cl4))
print('cl0')
print('flb0cl0',flb0cl0)
print('flb1cl0',flb1cl0)
print('flb2cl0',flb2cl0)
print('flb3cl0',flb3cl0)
print('cl1')
print('flb0cl1',flb0cl1)
print('flb1cl1',flb1cl1)
print('flb2cl1',flb2cl1)
print('flb3cl1',flb3cl1)
print('cl2')
print('flb0cl2',flb0cl2)
print('flb1cl2',flb1cl2)
print('flb2cl2',flb2cl2)
print('flb3cl2',flb3cl2)
print('cl3')
print('flb0cl3',flb0cl3)
print('flb1cl3',flb1cl3)
print('flb2cl3',flb2cl3)
print('flb3cl3',flb3cl3)
print('cl4')
print('flb0cl4',flb0cl4)
print('flb1cl4',flb1cl4)
print('flb2cl4',flb2cl4)
print('flb3cl4',flb3cl4)

#P-K (5-Check MSE and creat sub-dataset 1)
df=dfff[dfff['clusters']!=0]#,#4
df=df[df['clusters']!=2]#,#4
df=df[df['clusters']!=4]#,#4
df=df
dforg=pd.DataFrame(dfff.iloc[:,:5],index=df.index,columns=dfff.iloc[:,:5].columns)##2

#...Sil...
SC.append(means_lst[0])#,#4
SC.append(means_lst[2])#,#4
SC.append(means_lst[4])#,#4

#P-K (6-PCA)
X1=dforg.iloc[:,:5]##
p1=PCA()
p1.fit(X1)
W1=p1.components_.T
y1=p1.fit_transform(X1)
yhat1=X1.dot(W1)
yhat1.columns
#P-K (7-EV)
w1=pd.DataFrame(W1[:,:5],index=X1.columns[:])##
EV1=pd.DataFrame(p1.explained_variance_ratio_,index=np.arange(5)+1,columns=['Explained Variability'])##
Sev1=np.cumsum(p1.explained_variance_ratio_)
print(w1)
print(EV1)
print(Sev1)
#P-K (8-Check EV and reduce dimension)
X1=yhat1.iloc[:,:4]###
X1=pd.DataFrame(X1)
X1.columns
#P-K (9-Run K-means)
kmeans2 = KMeans(n_clusters=2).fit(X1)
labels3=kmeans2.labels_

#Sil
n_clusters=2
sample_silhouette_values = metrics.silhouette_samples(X1, labels3)
means_lst = []
for label in range(n_clusters):
    means_lst.append(sample_silhouette_values[labels3 == label].mean())
means_lst
SCF=means_lst+SC

#...K-means
X1['clusters']=kmeans2.labels_
print(X1.groupby('clusters').agg('mean'))
h0=X1[X1['clusters']==0]
h0=np.array(h0.iloc[:,:4])###
h1=X1[X1['clusters']==1]
h1=np.array(h1.iloc[:,:4])###
#MSE for each cluster
print('MSE n0',MSE(h0))
print('MSE n1',MSE(h1))
#SSE/N
print('WSSE kol fml',(SSE(h0)+SSE(h1)+SSE(c4)+SSE(c2)+SSE(c0))/dff.shape[0])
#Average MSE
print('AVMSE kol fml',(MSE(h0)+MSE(h1))/2)
#Sil Finall
print('SCFmean',np.mean(SCF))
#Interpretability
cll=X1['clusters']
cll0=cll.index[cll==0]
cll1=cll.index[cll==1]
dflbb=pd.DataFrame(dff3,index=df.index,columns=dff3.columns)
lbb=dflbb.iloc[:,6]
lbb0=lbb.index[lbb==0]
lbb1=lbb.index[lbb==1]
lbb2=lbb.index[lbb==2]
lbb3=lbb.index[lbb==3]
lbbmap = pd.DataFrame()
lbbmap['df_index'] = df.index.values
lbbmap['cll'] = kmeans2.labels_
lbbmap['lbb'] = lb
flbb0cll0=(len((lbbmap.loc[(lbbmap['lbb']==0)&(lbbmap['cll']==0)])))/(len(cll0))
flbb1cll0=(len((lbbmap.loc[(lbbmap['lbb']==1)&(lbbmap['cll']==0)])))/(len(cll0))
flbb2cll0=(len((lbbmap.loc[(lbbmap['lbb']==2)&(lbbmap['cll']==0)])))/(len(cll0))
flbb3cll0=(len((lbbmap.loc[(lbbmap['lbb']==3)&(lbbmap['cll']==0)])))/(len(cll0))

flbb0cll1=(len((lbbmap.loc[(lbbmap['lbb']==0)&(lbbmap['cll']==1)])))/(len(cll1))
flbb1cll1=(len((lbbmap.loc[(lbbmap['lbb']==1)&(lbbmap['cll']==1)])))/(len(cll1))
flbb2cll1=(len((lbbmap.loc[(lbbmap['lbb']==2)&(lbbmap['cll']==1)])))/(len(cll1))
flbb3cll1=(len((lbbmap.loc[(lbbmap['lbb']==3)&(lbbmap['cll']==1)])))/(len(cll1))

print('cl0')
print('flbb0cll0',flbb0cll0)
print('flbb1cll0',flbb1cll0)
print('flbb2cll0',flbb2cll0)
print('flbb3cll0',flbb3cll0)
print('cl1')
print('flbb0cll1',flbb0cll1)
print('flbb1cll1',flbb1cll1)
print('flbb2cll1',flbb2cll1)
print('flbb3cll1',flbb3cll1)

stop = timeit.default_timer()
print('Time: ', stop - start)
