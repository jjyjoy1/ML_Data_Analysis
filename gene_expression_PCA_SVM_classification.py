import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Train_Data = pd.read_csv("../data_set_ALL_AML_train.csv")
Test_Data = pd.read_csv("../data_set_ALL_AML_independent.csv")
labels = pd.read_csv("../actual.csv", index_col = 'patient')

Train_Data.head()
Test_Data.head()

print(Train_Data.isna().sum().max())
print(Test_Data.isna().sum().max())

cols = [col for col in Test_Data.columns if 'call' in col]
test = Test_Data.drop(cols, axis=1)
cols = [col for col in Train_Data.columns if 'call' in col]
train = Train_Data.drop(cols, axis=1)

patients = [str(i) for i in range(1, 73, 1)]
df_all = pd.concat([train, test], axis = 1)[patients]
df_all = df_all.T

df_all["patient"] = pd.to_numeric(patients)
labels["cancer"]= pd.get_dummies(labels.cancer, drop_first=True)

Data = pd.merge(df_all, labels, on="patient")
Data.head()
Data.columns = Data.columns.map(str)
Data.to_csv("./gene_expression.csv")

X, y = Data.drop(columns=["cancer"]), Data["cancer"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
total=sum(pca.explained_variance_)
k=0
current_variance=0
while current_variance/total < 0.90:
    current_variance += pca.explained_variance_[k]
    k=k+1
    

from sklearn.decomposition import PCA
pca = PCA(n_components = 38)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
cum_sum = pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100
plt.bar(range(38), cum_sum)
plt.ylabel("Cumulative Explained Variance")
plt.xlabel("Principal Components")
plt.title("Around 90% of variance is explained by the First 38 columns ")

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

search = GridSearchCV(SVC(), parameters, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

best_parameters = search.best_estimator_

model = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)


model.fit(X_train, y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
print('Accuracy Score:',round(accuracy_score(y_test, y_pred),2))
#confusion matrix
cm = confusion_matrix(y_test, y_pred)

class_names=[1,2,3]
fig, ax = plt.subplots()

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
class_names=['ALL', 'AML']
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

########

#https://medium.com/leukemiaairesearch/complexity-reduction-techniques-with-gene-expression-data-961491979bc8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

data = pd.read_table("201028_GSE122505_Leukemia_clean.txt", sep = "\t")

data.disease.value_counts()

data["disease"] = np.where(data["disease"] == "Diabetes_Type_I" , "Diabetes", data["disease"])
data["disease"] = np.where(data["disease"] == "Diabetes_Type_II" , "Diabetes", data["disease"])
other = ['CML','clinically_isolated_syndrome', 'MDS', 'DS_transient_myeloproliferative_disorder']
data = data[~data.disease.isin(other)]
data.shape
target = data["disease"]
df = data.drop("disease", axis=1)
df = df.drop("GSM", axis=1)
df = df.drop("FAB", axis=1)
df.shape
target.value_counts()

df = df.drop(df.var()[(df.var() < 0.3)].index, axis=1)
from scipy.stats import zscore
df = df.apply(zscore)
df.shape

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X = pca.fit(df).transform(df)
print(pca.explained_variance_ratio_)

#pca = PCA()
#X = pca.fit(df).transform(df)
#total=sum(pca.explained_variance_)
#k=0
#current_variance=0
#while current_variance/total < 0.90:
    current_variance += pca.explained_variance_[k]
    k=k+1
    



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(target)
y_lan = le.transform(target)
pca_df = pd.DataFrame(columns = [“x”, “y”, “name”, “label”])
pca_df[“PCA1”] = X[:, 0]
pca_df[“PCA2”] = X[:, 1]
pca_df[“Disease”] = target
pca_df[“label”] = y_lan
sns.set(style=”whitegrid”, palette=”muted”)
#sns.set_theme(style=”whitegrid”)
ax = sns.scatterplot(x=”PCA1", y=”PCA2", hue=”Disease”, data=pca_df)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#saving the graph in png or jpeg
#plt.savefig(“GSE122505_Leukemia_PCA.pdf”, dpi = 300)
#plt.savefig(“GSE122505_Leukemia_PCA.png”)
#pca_df.to_csv(“GSE122505_Leukemia_PCA.csv”)


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=48.0, learning_rate=200.0, n_iter=2000 )
X = tsne.fit_transform(df, y_lan)
tsne_df = pd.DataFrame(columns = ["x", "y", "name", "label"])
tsne_df["tSNE1"] = X[:, 0]
tsne_df["tSNE2"] = X[:, 1]
tsne_df["Disease"] = target
tsne_df["label"] = y_lan

sns.set(style="whitegrid", palette="muted")
ax = sns.scatterplot(x="tSNE1", y="tSNE2", hue="Disease",  data=tsne_df)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig("GSE122505_Leukemia_tSNE.pdf", dpi = 300)
#plt.savefig("GSE122505_Leukemia_tSNE.png")
#tsne_df.to_csv("GSE122505_Leukemia_tSNE.csv")

pca = PCA(n_components=50)
X_pca = pca.fit(df).transform(df)

tsne = TSNE(n_components=2, perplexity=48.0, learning_rate=200.0, n_iter=2000 )
X = tsne.fit_transform(X_pca, y_lan)
tsne_df = pd.DataFrame(columns = ["x", "y", "name", "label"])
tsne_df["tSNE1"] = X[:, 0]
tsne_df["tSNE2"] = X[:, 1]
tsne_df["Disease"] = target
tsne_df["label"] = y_lan

sns.set(style="whitegrid", palette="muted")
ax = sns.scatterplot(x="tSNE1", y="tSNE2", hue="Disease",  data=tsne_df)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig("GSE122505_Leukemia_PCA_tSNE.pdf", dpi = 300)
#plt.savefig("GSE122505_Leukemia_PCA_tSNE.png")
#tsne_df.to_csv("GSE122505_Leukemia_PCA_tSNE.csv")

import umap
reducer = umap.UMAP(n_neighbors =  100, min_dist= 0.2, metric ="euclidean")
X_umap = reducer.fit_transform(df)
umap_df = pd.DataFrame(columns = ["x", "y", "name", "label"])
umap_df["UMAP1"] = X_umap[:, 0]
umap_df["UMAP2"] = X_umap[:, 1]
umap_df["Disease"] = target
umap_df["label"] = y_lan

sns.set(style="whitegrid", palette="muted")
ax = sns.scatterplot(x="UMAP1", y="UMAP2", hue="Disease",  data=umap_df)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig("GSE122505_Leukemia_UMAP.pdf", dpi = 300)
#plt.savefig("GSE122505_Leukemia_UMAP.png")
#ctrl_df.to_csv("GSE122505_Leukemia_UMAP.csv")

pca = PCA(n_components=50)
X_pca = pca.fit(df).transform(df)

reducer = umap.UMAP(n_neighbors =  100, min_dist= 0.2, metric ="euclidean")
X = reducer.fit_transform(X_pca)

umap_df = pd.DataFrame(columns = ["x", "y", "name", "label"])
umap_df["UMAP1"] = X[:, 0]
umap_df["UMAP2"] = X[:, 1]
umap_df["Disease"] = target
umap_df["label"] = y_lan

sns.set(style="whitegrid", palette="muted")
ax = sns.scatterplot(x="UMAP1", y="UMAP2", hue="Disease",  data=umap_df)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig("GSE122505_Leukemia_PCA_UMAP.pdf", dpi = 300)
#plt.savefig("GSE122505_Leukemia_PCA_UMAP.png")
#ctrl_df.to_csv("GSE122505_Leukemia_PCA_UMAP.csv")


umap_df["ANXA1"] = df["ANXA1"]
sns.set(style="whitegrid", palette="muted")
ax = sns.scatterplot(x="UMAP1", y="UMAP2", hue="ANXA1",  data=umap_df)
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#PCA and then kmeans clustering
pca = PCA(n_components=4)
X_pca = pca.fit(df).transform(df)

print (pca.explained_variance_ratio_)

K=[]
for i in range(1,31):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 111)
    kmeans_pca.fit(X_pca)
    K.append(kmeans_pca.inertia_)
    
plt.figure(figsize = (10,8))
plt.plot(range(1,31), K, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('K')
plt.title('K-means with PCA Clustering')
plt.show()

#kmeans_pca = KMeans(n_clusters=5, init = 'k-means++', random_state = 111)
kmeans_pca = KMeans(n_clusters=7, init = 'k-means++', random_state = 111)
kmeans_pca.fit(X_pca)
df_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(X_pca)], axis=1)

#df_pca_kmeans.columns.values[-4:] = ['Comp 1','Comp 2','Comp 3','Comp 4']
df_pca_kmeans.columns.values[-4:] = ['Comp 1','Comp 2','Comp 3','Comp 4']

df_pca_kmeans["X Kmeans PCA"] = kmeans_pca.labels_

df_pca_kmeans["X_data"] =df_pca_kmeans["X Kmeans PCA"].map({0:'first', 1:'second', 2:'third',3:'fourth', 4:'five', 5:'six', 6:'seven'})

x_axis = df_pca_kmeans['Comp 2']
y_axis = df_pca_kmeans['Comp 1']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis,y_axis, hue = df_pca_kmeans['X_data'], palette = ['g','r','c','m','b','k','y'])
plt.title("Clusters by PCA Components")
plt.show()


#PCA 3D picture
pca = PCA(n_components=3)
components = pca.fit_transform(X)

# 3D scatterplot
fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=clusters, size=0.1*np.ones(len(X)), opacity = 1,
    title='PCA plot in 3D',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'},
    width=800, height=500
)
fig.show()


# t-SNE
tsne = TSNE(n_components=3)
components_tsne = tsne.fit_transform(X)

# 3D scatterplot
fig = px.scatter_3d(
    components_tsne, x=0, y=1, z=2, color=clusters, size=0.1*np.ones(len(X)), opacity = 1,
    title='t-SNE plot in 3D',
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=800, height=500
)
fig.show()

# UMAP
um = umap.UMAP(n_components=3)
components_umap = um.fit_transform(X)

# 3D scatterplot
fig = px.scatter_3d(
    components_umap, x=0, y=1, z=2, color=clusters, size=0.1*np.ones(len(X)), opacity = 1,
    title='UMAP plot in 3D',
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=800, height=500
)
fig.show()
