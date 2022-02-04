import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns


Cluster_1_csv = pd.read_csv('Spindle_AP_ratio_1.csv')
Cluster_3_csv = pd.read_csv('Spindle_AP_ratio_3.csv')

Cl1_APs = Cluster_1_csv['Spindle AP ratio'].tolist()
Cl1_APs = np.array(Cl1_APs)
Cl1_APs = Cl1_APs[np.logical_not(np.isnan(Cl1_APs))]

Cl1_Bursts = Cluster_1_csv['Spindle Burst ratio'].tolist()
Cl1_Bursts = np.array(Cl1_Bursts)
Cl1_Bursts = Cl1_Bursts[np.logical_not(np.isnan(Cl1_Bursts))]

Cl3_APs = Cluster_3_csv['Spindle AP ratio'].tolist()
Cl3_APs = np.array(Cl3_APs)
Cl3_APs = Cl3_APs[np.logical_not(np.isnan(Cl3_APs))]
Cl3_Bursts = Cluster_3_csv['Spindle Burst ratio'].tolist()
Cl3_Bursts = np.array(Cl3_Bursts)
Cl3_Bursts = Cl3_Bursts[np.logical_not(np.isnan(Cl3_Bursts))]


print(Cl3_APs)
stat_APs, pval_APs = stats.ttest_ind(Cl1_APs, Cl3_APs, axis=0, equal_var = False)
stat_Bursts, pval_Bursts = stats.ttest_ind(Cl1_Bursts, Cl3_Bursts, axis=0, equal_var = False)

print('Spikes p-val: ',pval_APs)
print('Bursts p-val: ',pval_Bursts)


Cl1_spikes_label = ['Cl1 Spikes']*len(Cl1_APs)
Cl1_bursts_label = ['Cl1 Bursts']*len(Cl1_Bursts)
Cl3_spikes_label = ['Cl3 Spikes']*len(Cl3_APs)
Cl3_bursts_label = ['Cl3 Bursts']*len(Cl3_Bursts)
Labels_str_spikes = Cl1_spikes_label+Cl3_spikes_label
Labels_str_bursts = Cl1_bursts_label+Cl3_bursts_label
Dat_points_spikes = np.append(Cl1_APs,Cl3_APs)
Dat_points_bursts = np.append(Cl1_Bursts,Cl3_Bursts)

sns.boxplot(Labels_str_spikes,Dat_points_spikes,whis=[0, 100], width=.6, palette="vlag")
sns.stripplot(Labels_str_spikes,Dat_points_spikes,size=4, color=".3", linewidth=0)
plt.ylabel("Relative spike rate")
plt.title("Spike rate during spindles")
#sns.boxplot(B_x,C1_NREM_AUC)
plt.show()

sns.boxplot(Labels_str_bursts,Dat_points_bursts,whis=[0, 100], width=.6, palette="vlag")
sns.stripplot(Labels_str_bursts,Dat_points_bursts,size=4, color=".3", linewidth=0)
plt.ylabel("Relative rate")
plt.title("Burst rate during spindles")
#sns.boxplot(B_x,C1_NREM_AUC)
plt.show()
