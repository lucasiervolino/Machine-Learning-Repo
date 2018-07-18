#KMeans used for clustering data
#aggregates new features
from sklearn.cluster import KMeans

flist_kmeans = []
for ncl in range(2,5):
    cls = KMeans(n_clusters=ncl)
    cls.fit_predict(x_train[flist].values)
    x_train['kmeans_cluster_'+str(ncl)] = cls.predict(x_train[flist].values)
    x_test['kmeans_cluster_'+str(ncl)] = cls.predict(x_test[flist].values)
    flist_kmeans.append('kmeans_cluster_'+str(ncl))
print(flist_kmeans)



