# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Open csv file with data generated from script Feature_analysis
with open('FeaturesArray_140520_1.csv', newline='') as csvfile:
    # Extract contents of csv file to a list
    data = list(csv.reader(csvfile))
    # Convert list to numpy array
    Dat_np = np.array(data)

    # Extract the list of feature names  
    Features = Dat_np[0,[1,2,3,5,6,7,9,10,11,12,13,14]]
    # Extract all the features
    Dat_Red = Dat_np[1:,[1,2,3,5,6,7,9,10,11,12,13,14]]
    #print('Data has type ',type(Dat_np))

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(Dat_Red)
# Create a PCA instance: pca
pca = PCA(n_components=12)
# Perform the PCA on the data array X_std
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
# explained_variance_ratio_ defines how much variance is covered by each principal component
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.show()
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
print('PCA components size = ', np.shape(PCA_components))
#plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
#plt.xlabel('PCA 1')
#plt.ylabel('PCA 2')
#plt.show()



ks = range(1, 10)
inertias = []
print('PCA Components',PCA_components.iloc[:,0])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(PCA_components.iloc[:,0],PCA_components.iloc[:,1],PCA_components.iloc[:,2],marker = 'o')
plt.show()
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
	elif model.labels_[i]==4:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=.1, color='black')
	elif model.labels_[i]==5:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=.1, color='purple')
	elif model.labels_[i]==6:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=.1, color='cyan')
	elif model.labels_[i]==7:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=.1, color='yellow')
	elif model.labels_[i]==8:
		plt.scatter(PCA_components[0][i], PCA_components[1][i], alpha=.1, color='red')			


print('Cluster 1 size =',np.count_nonzero(model.labels_ == 0))
print('Cluster 2 size =',np.count_nonzero(model.labels_ == 1))
print('Cluster 3 size =',np.count_nonzero(model.labels_ == 2))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.show()
print('PCA components type and shape =',type(pca.components_),', ',np.shape(pca.components_))
plt.matshow(abs(pca.components_[0:2,:]),cmap='viridis')
print('Components = ',np.shape(pca.components_))
plt.yticks([0,1],['1st Comp','2nd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(Features)),Features,rotation=65,ha='left')
plt.tight_layout()
plt.show()# 

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

df = pd.DataFrame({"Clusters" : model.labels_})
df.to_csv("Cluster_labels.csv", index=False)
'''
df = pd.DataFrame({"Cl1" : Dat_np[Cl1,0]})
df.to_csv("CellName_cluster1_test.csv", index=False)


df = pd.DataFrame({"Cl2" : Dat_np[Cl2,0]})
df.to_csv("CellName_cluster2_test.csv", index=False)


df = pd.DataFrame({"Cl3" : Dat_np[Cl3,0]})
df.to_csv("CellName_cluster3_test.csv", index=False)
print(Dat_np[Cl1,0])
'''

'''
y = np.array(Spike_Stats_Mat)
from tempfile import TemporaryFile
Spindle_spike_prob = TemporaryFile()
np.savetxt('Spike_Stats.csv', y, delimiter=',', fmt='%s')
'''