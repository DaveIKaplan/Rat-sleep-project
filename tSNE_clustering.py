from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

with open('Cluster_labels.csv', newline='') as csvfile:
	Labels_PCA = list(csv.reader(csvfile))
	Labels_PCA = np.array(Labels_PCA)
	print('Labels PCA = ',np.shape(Labels_PCA[1:]))
# Open csv file with data generated from script Feature_analysis
with open('FeaturesArray_140520_1.csv', newline='') as csvfile:
    # Extract contents of csv file to a list
    data = list(csv.reader(csvfile))
    # Convert list to numpy array
    Dat_np = np.array(data)

    # Extract the list of feature names  
    Features = Dat_np[0,[1,2,3,5,6,7,9,10,11,12,13,14]]
    # Extract all the featureperplexity = 10s
    Dat_Red = Dat_np[1:,[1,2,3,5,6,7,9,10,11,12,13,14]]
    #print('Data has type ',type(Dat_np))

tsne = TSNE(perplexity = 30)
X_embedded = tsne.fit_transform(Dat_Red)

ks = range(1, 10)
inertias = []
print('X_embedded',np.shape(X_embedded))
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=3)
    
    # Fit model to samples
    # print(np.shape(PCA_components.iloc[:,:2]))
    model.fit(X_embedded)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)


#plt.subplot(211)
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

print('indices of clusters = ',model.labels_)
print(type(model.labels_))
#plt.subplot(212)
'''
for i in range(0,len(model.labels_)):
	if Labels_PCA[i]=='0':
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 10, alpha=1, color='green')
	elif Labels_PCA[i]=='1':
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 10, alpha=1, color='red')
	elif Labels_PCA[i]=='2':
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 10, alpha=1, color='blue')
	elif Labels_PCA[i]=='3':
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 10, alpha=1, color='orange')
'''
for i in range(0,len(model.labels_)):
	if model.labels_[i]==0:
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 11, alpha=1, color='green')#, edgecolors = 'none')
	elif model.labels_[i]==1:
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 11, alpha=1, color='red')#, edgecolors = 'none')
	elif model.labels_[i]==2:
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 11, alpha=1, color='blue')#, edgecolors = 'none')
	elif model.labels_[i]==3:
		plt.scatter(X_embedded[i,0], X_embedded[i,1], marker = 11, alpha=1, color='orange')#, edgcecolors = 'none')


#sns.scatterplot(X_embedded[:,0], X_embedded[:,1])#, hue=y, legend='full', palette=palette)
plt.title('Clusters following tSNE')
plt.show()

# Save a csv file with the filenames of each file from each cluster

Cl1 = np.where(model.labels_==0)
Cl2 = np.where(model.labels_==1)
Cl3 = np.where(model.labels_==2)
print('Cl1 type is ',type(Cl1))
print(Cl1[0])

Cl1 = Cl1[0]+1

Cl2 = Cl2[0]+1
Cl3 = Cl3[0]+1
print(Dat_np[Cl1,0])
print(Dat_np[Cl2,0])
print(Dat_np[Cl3,0])

df = pd.DataFrame({"Cl1" : Dat_np[Cl1,0]})
df.to_csv("tSNE_CellName_cluster1_test.csv", index=False)


df = pd.DataFrame({"Cl2" : Dat_np[Cl2,0]})
df.to_csv("tSNE_CellName_cluster2_test.csv", index=False)


df = pd.DataFrame({"Cl3" : Dat_np[Cl3,0]})
df.to_csv("tSNE_CellName_cluster3_test.csv", index=False)

'''
ks = range(1, 10)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=3)#k)
    
    # Fit model to samples
    # print(np.shape(PCA_components.iloc[:,:2]))
    model.fit(PCA_components.iloc[:,:2])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)


#plt.subplot(211)
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

print('indices of clusters = ',model.labels_)
print(type(model.labels_))
#plt.subplot(212)
for i in range(0,len(model.labels_)):
	if model.labels_[i]==0:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=1, color='green')
	elif model.labels_[i]==1:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=1, color='red')
	elif model.labels_[i]==2:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=.1, color='blue')
	elif model.labels_[i]==3:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=.1, color='orange')

sns.scatterplot(X_embedded[:,0], X_embedded[:,1])#, hue=y, legend='full', palette=palette)
plt.show()

'''