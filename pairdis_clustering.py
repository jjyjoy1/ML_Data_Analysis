#This script was prepared for genome size sequences pairwise distance clustering
#Distance matirx was the output sourmash
#sourmash sketch dna *.fna -p k=21,k=31,k=51 (the output will be *fna.sig)
#or gunzip -c data/GCF*.fna.gz | sourmash sketch dna - -o out.sig

#sourmash compare *sig --output sourmash.matrix --ksize 21
#sourmash plot sourmash.matrix --csv smatrix.csv
#smatrix.csv is pairwise distance for clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


df = pd.read_csv("smatrix.csv")
df.index=df.columns.to_list()
ndf = df.to_numpy()

#First explore the potential variants distance in whole genome sequences

#All distance range
all_genomes = [x for x in df.index]
all_genomes_centroid_dist = df[all_genomes].apply(min,1)
#review the distance
all_genomes_centroid_dist.hist(bins=100)

#Using PCA and kmeans to explore the potential structure
n_digits = 20
pca = PCA().fit(df)
reduced_data = PCA(n_components=0.95).fit_transform(df)
#reduced_data = PCA(n_components=2).fit_transform(df)

kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=100)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')


plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the DNA sequence dataset \n(PCA-reduced distance matrix)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

from scipy.spatial import distance

grouped = df.groupby(kmeans.labels_)
centroids = []
for name,group in grouped:
    closest_to_centroid = pd.DataFrame(reduced_data).groupby(kmeans.labels_).get_group(name).apply(
        lambda x: distance.euclidean(x,kmeans.cluster_centers_[name]), axis=1).sort_values().index[0]
    print("Number of sequences in group: {}".format(len(group)))
    #Reduce the distance matrix to be square within the group
    reduced_group = group[group.index]
    print("Max distance within group: {}".format(max(reduced_group.apply(max))))
    closest_id = df.index[closest_to_centroid]
    print("ID closest to centroid (Euclidean): {}".format(closest_id))
    centroids.append(closest_id)
    
#    print("Furthest within-group P distance distances to centroids ID:")
#    print(reduced_group[closest_id].sort_values(ascending=False)[0:2])
#    print()
print("Centroids: ", centroids)
centroid_dist = df[centroids].apply(min,1)
centroid_dist.hist(bins=100)
#We can use the select genome

#However, we still need to pick the number of clusters that minimizes the number of sequences with >30% divergence

divergent_seqs = []
pca = PCA().fit(df)
reduced_data = PCA(n_components=0.95).fit_transform(df)

for i in range(3,100):
    n_digits = i #number of clusters
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)
    grouped = df.groupby(kmeans.labels_)
    centroids = []
    for name,group in grouped:
        #Find the sample that is closest to the centroid. This is a pd Dataframe row index.
        closest_to_centroid = pd.DataFrame(reduced_data).groupby(kmeans.labels_).get_group(name).apply(
            lambda x: distance.euclidean(x,kmeans.cluster_centers_[name]), axis=1).sort_values().index[0]
        closest_id = df.index[closest_to_centroid]
        centroids.append(closest_id)
    #print("Centroids: ", centroids)
    centroid_dist = df[centroids].apply(min,1)
    num_over_25 = len(centroid_dist[centroid_dist > 0.30])
    divergent_seqs.append((i,num_over_25))
    #centroid_dist.hist(bins=100)
    #plt.title("Minimum Distance to Centroid, {} Clusters\n Gene {}".format(n_digits,gene))
    #print("\n\n Distances > 25%:")
    #print("\nNumber of Distances > 25%: {}".format(len(centroid_dist[centroid_dist > 0.25])))

divergent_seqs_df = pd.DataFrame(divergent_seqs,columns=["NumClusters","NumDivergent"])
plt.plot(divergent_seqs_df.NumClusters,divergent_seqs_df.NumDivergent,'-o')
plt.xlabel("Number of clusters")
plt.title("Number of sequences with > 30% divergence from any centroid")



#Another option for assigning sequences to clusters is spectral clustering.
import numpy as np
import scipy as sp
from sklearn.cluster import spectral_clustering

similarity = np.exp(-2 * df / df.std()).to_numpy()

labels = spectral_clustering(similarity,n_clusters=6,assign_labels = 'discretize')
colormap = np.array(["r","g","b","w","purple","orange","brown","lightblue"])
plt.scatter(reduced_data[:, 0], reduced_data[:, 1],c=colormap[labels])
plt.xticks(())
plt.yticks(())
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(ndf, 'ward')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()

Z = linkage(ndf, 'single')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()

from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist

df1 = pd.DataFrame()
Z = single(ndf)
df1['c']= pd.DataFrame(fcluster(Z, 0.35, criterion='distance'))
df1.index=df.index.to_list()


#
def kMedoids(D, k, tmax=1000):
    # determine dimensions of distance matrix D
    m, n = D.shape
    # randomly initialize an array of k medoid indices
    M = np.sort(np.random.choice(n, k))
    # create a copy of the array of medoid indices
    Mnew = np.copy(M)
    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    # return results
    return M, C


for t in range(100):
    J = np.argmin(ndf[:,M], axis=1)
    for kappa in range(8):
        C[kappa] = np.where(J==kappa)[0]
    for kappa in range(8):
        J = np.mean(ndf[np.ix_(C[kappa],C[kappa])],axis=1)
        j = np.argmin(J)
        Mnew[kappa] = C[kappa][j]
    np.sort(Mnew)
    if np.array_equal(M, Mnew):
        break
    M = np.copy(Mnew)
else:
    # final update of cluster memberships
    J = np.argmin(ndf[:,M], axis=1)
    for kappa in range(8):
        C[kappa] = np.where(J==kappa)[0]
