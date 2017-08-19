import pandas as pd; import numpy as np
k = 5
df = pd.read_csv('C:/Users/bodhisattva_2/Dropbox/MCC2_CurrAccnts_TSeries_M.csv')
row_count = len(df)
df['cluster'] = pd.Series(np.full(row_count, 0), df.index)
clusters = []
for j in range(0, k):
	centroid = df.quantile((1/k)*(j+1)).values[0]
	clusters.append(centroid)
changed_assign = True
while (changed_assign):
	changed_assign = False
	for i in range(0, row_count):
		obsv = df['trans_amt'].loc[i]
		curr_clust = df['cluster'].loc[i]
		min = (obsv-clusters[0])**2
		clust = 0
		for j in range(1, k):
			test_ssd = (obsv-clusters[j])**2
			if (test_ssd < min):
				min = test_ssd
				clust = j
		if (curr_clust != clust):
			changed_assign = True
			df['cluster'].loc[i] = clust
	new_centroids = df['trans_amt'].groupby(df['cluster']).mean()
	for j in new_centroids.index:
		j = int(j)
		clusters[j] = new_centroids.loc[j]
df.to_csv('C:/Users/bodhisattva_2/Dropbox/MCC2_CurrAccnts_TSeries_M_kMeans5.csv')
