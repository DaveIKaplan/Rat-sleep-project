import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as tickr
import scipy.io as sio
from scipy import signal
import random as rnd 
from scipy.stats import spearmanr
import os.path
from os import path
from matplotlib.lines import Line2D
from matplotlib.widgets import Button as btn
from tkinter import *
from tkinter import simpledialog
import csv
from tempfile import TemporaryFile
import pandas as pd
import random as rnd
import neo
import quantities

import mne
from mne import io
from mne.time_frequency import tfr_stockwell


for Cluster_no in [3, 1]:

	# Variables
	dt = 4e-5
	srate=1000
	Win_T = np.arange(-2+dt,2-dt,dt)
	smoothWindow=np.zeros((51,))
	x=np.arange(-0.025,0.026,0.001)

	y = 0.001* ((1 / (0.005*np.sqrt(2 * np.pi))) * np.exp((-1/2)*((x/0.005) ** 2)))
	smoothWindow=y#0.02
	#Cluster_no = 3
	'''
	Code for generating the spectrogram
	'''
	# Fixed parameters 
	Win_T_down = Win_T[0:-26:25] # Downsampled time window for 4s around spike
	Win_T_Down = Win_T_down
	num_frex = 65
	range_cycles = [4  ,14]
	min_freq = 0.5#0.1
	max_freq = 100#45
	frex = np.linspace(min_freq,max_freq,num = num_frex)
	t_wav  = np.arange(-2,(2-(1/srate)),(1/srate))
	nCycs = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[-1]),num = num_frex)
	half_wave = (len(t_wav)-1)/2
	QW_btLFP_AUC_all = []
	QW_btLFP_AUC_ctrl_all = []
	NREM_btLFP_AUC_all = []
	NREM_btLFP_AUC_ctrl_all = []
	#frex = np.logspace(np.log10(min_freq),np.log10(max_freq),num = num_frex)
	#s = logspace(log10(range_cycles(1)),log10(range_cycles(end)),num_frex) ./ (2*pi*frex);

	def spectrogram(y,t_win):
		

		# FFT parameters
		nKern = len(t_wav)
		nData = len(y)
		nConv = nKern+nData-1
		# Convert data to frequency domain
		dataX   = np.fft.fft( y ,nConv )
		tf = np.zeros((num_frex,len(y)-1)) #np.zeros((num_frex,len(t_win)-1))
		#tf = np.zeros((num_frex,nConv-1))

		# Loop through each frequency.
		for fi in range(0,num_frex):
			s = nCycs[fi]/(2*np.pi*frex[fi]);
			# Wavelet function	
			wavelet = np.exp(2*complex(0,1)*np.pi*frex[fi]*t_wav) * np.exp(-t_wav**2/(2*s**2))
			# Wavelet function in the frequency domain
			waveletX = np.fft.fft(wavelet,nConv);
			waveR = [wr.real for wr in wavelet]
			waveI = [wi.imag for wi in wavelet]
			#plt.plot(t_wav,waveR)
			#plt.show()

			# Multiply the fourier transform of the wavelet by the fourier transform of the data
			# then compute the inverse fourier transform to convert back to the time domain.
			As = np.fft.ifft(np.multiply(waveletX,dataX),nConv)
			As = As[int(half_wave)+1:-int(half_wave)]
			#print(np.shape(tf[fi,:]))
			#print(np.shape(As))
			tf[fi,:] = abs(As)**2

		# for i in range(1,num_frex):
	   	#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
		#	tf[i,:] = 100 * np.divide(np.subtract(tf[i,:],np.mean(tf[i,:])), np.mean(tf[i,:]))
		Spect_Out = [tf, frex, t_win[0:-1]]
		return Spect_Out

	# Import cluster
	Cluster = ('CellName_cluster{}'.format(Cluster_no))
	Cluster_csv = (Cluster + '.csv')
	with open(Cluster_csv, 'r') as file: #"CellName_cluster1.csv"
		reader = csv.reader(file)
		CellNames = []
		for row in reader:
			#print(row[0])
			CellName = row[0]
			CellNames.append(CellName)


	NREM_i = 0
	QW_i = 0
	Summary_Matrix_NREM = np.zeros((num_frex,len(Win_T[0:-26:25]),2))
	Summary_Matrix_QW = np.zeros((num_frex,len(Win_T[0:-26:25]),2))

	Summary_Matrix_NREM_ctrl = np.zeros((num_frex,len(Win_T[0:-26:25]),2))
	Summary_Matrix_QW_ctrl = np.zeros((num_frex,len(Win_T[0:-26:25]),2))
	#Spike_spectra = np.zeros((num_frex,len(Win_T[0:-26:25]),int(len(QW_Spikes_all_i)/2)))
	'''
	Start looping through cells 
	'''
	for i in range(1,len(CellNames)):
		#try:
		print(CellNames[i])
		Filename = CellNames[i]
		Cell = ('C:/Users/dkapl/Dropbox/Matthew Larkum/Matlab backup 30052017/MATLAB/Sleep project/SQL data/Extracted Data/MySQLdata_{}.mat'.format(Filename))
		print(Filename[2])
		# create a reader
		if Filename[2] == '6' or Filename[2] == '7' or Filename[2] == '8' or Filename[2] == '9':
			Cell_smr = ('C:/Users/dkapl/Dropbox/Matthew Larkum/Guy Doron/GuyJulie/JulieScored/{}.smr'.format(Filename))
		else:
			Cell_smr = ('C:/Users/dkapl/Dropbox/Matthew Larkum/Guy Doron/GuyJulie/JulieScored/{}Score.smr'.format(Filename))
		
		reader = neo.io.Spike2IO(filename=Cell_smr,try_signal_grouping = False)
		# read the block
		data = reader.read(lazy = False)[0]
		#print(neo.io.Spike2IO.readable_objects)
		Voltage_1 =	data.segments[0].analogsignals[1].rescale('mV').magnitude
		#print(len(Voltage_1))
		Time_smr = np.arange(0,len(Voltage_1)*4e-5,4e-5)
		Data = sio.loadmat(Cell)
		Time = Data['t']
		Time = np.squeeze(Time)
		LFP = Data['lfp']

		
		
		#LFP = np.sin(2*np.pi*0.5*Time)+0.5*np.sin(2*np.pi*20*Time)
		LFP = np.squeeze(Voltage_1)
		#print('shape voltage = ',np.shape(Voltage_1))
		#print('LFP =',[LFP[8:11],Voltage_1[8:11]])
		#print('Time =',[Time[8:11],Time_smr[8:11]])


		EEG_FF = Data['eeg1']
		EEG_FP = Data['eeg2']
		
		if Filename[2] != '3':
			EEG_FF = Data['eeg2']
			EEG_FP = Data['eeg1']


		EEG_FF = np.squeeze(EEG_FF)
		EEG_FP = np.squeeze(EEG_FP)
		


		Cell2 = ('C:/Users/dkapl/Dropbox/Matthew Larkum/Matlab backup 30052017/MATLAB/Sleep project/SQL data/newdata.mat')
		Data2 = sio.loadmat(Cell2)
		Data2 = Data2['AllCells']
		Cellnames = Data2['CellName']
		Cellnames = Cellnames[0] 

		SPIKES_cell = Data2['SpikeTimes']
		SPIKES_cell = SPIKES_cell[0] 

		loc1=(np.where(Cellnames == Filename))
		Spikes = SPIKES_cell[loc1][0]
		Spikes = np.squeeze(Spikes)

		Bursts = Data['BURSTS']
		Bursts = np.squeeze(Bursts)

		# States start and end times
		NREMStartT = Data['NREMStartTimes']
		#print(np.shape(NREMStartT))
		NREMEndT = Data['NREMEndTimes']
		#print(np.shape(NREMEndT)) 
		REMStartT = Data['REMStartTimes']
		#print(np.shape(REMStartT))
		REMEndT = Data['REMEndTimes'] 
		#print(np.shape(REMEndT))
		QWStartT = Data['QWStartTimes']
		#print(np.shape(QWStartT))
		QWEndT = Data['QWEndTimes']              
		#print(np.shape(QWEndT))
		#print('Start view', QWStartT)
		#print('End view', QWEndT)
		if NREMStartT[0]>NREMEndT[0]:
			NREMEndT = NREMEndT[1:]
		if NREMEndT[-1]<NREMStartT[-1]:
			NREMStartT = NREMStartT[:-1]
		#if REMStartT[0]>REMEndT[0]:
		#	REMEndT = REMEndT[1:]
		#if REMEndT[-1]<REMStartT[-1]:
		#	REMStartT = REMStartT[:-1]
		if QWStartT[0]>QWEndT[0]:
			QWEndT = QWEndT[1:]
		if QWEndT[-1]<QWStartT[-1]:
			QWStartT = QWStartT[:-1]

		'''
		Iterate through QW episodes and find BURSTS from each state
		-----------------------------------------------------------
		'''

		'''
		Iterate through QW episodes and find BURSTS from each state
		-----------------------------------------------------------
		'''
		# Find all bursts during QW and add them to the vector QW_Bursts_all
		QW_Bursts_all = []
		QW_Bursts_ctrl_all = []
		try:
			for j in range(0,len(QWStartT)):
				QW_Bursts = Bursts[np.where(np.logical_and(Bursts>=QWStartT[j], Bursts<=QWEndT[j]))]
				# Declare a vector of zeros to load with the control timesteps
				QW_Bursts_ctrl = np.zeros(len(QW_Bursts),)
				# Fill that vector with random timepoints during this QW window
				for k in range(0,len(QW_Bursts)-1):
					QW_Bursts_ctrl[k] = rnd.uniform(QWStartT[j],QWEndT[j])
				# Add the burst times from this QW episode to the array of bursts for the entire recoring 
				QW_Bursts_all.extend(QW_Bursts)
				# Do the same for the control timestamps
				QW_Bursts_ctrl_all.extend(QW_Bursts_ctrl)
			print('QW_Bursts size: ',np.shape(QW_Bursts_all),np.shape(QW_Bursts_ctrl_all))
			print('QW_Bursts: ',QW_Bursts_all[-1],QW_Bursts_ctrl_all[-1])
			# Conver burst times to burst indexes
			QW_Bursts_all_i = [int(x/dt) for x in QW_Bursts_all]
			QW_Bursts_ctrl_all_i = [int(x/dt) for x in QW_Bursts_ctrl_all]
			
			# Define the 3D matrix to contain the spectrograms for each burst (Freqs x Time x Bursts) 
			Burst_spectra = np.zeros((num_frex,len(Win_T[0:-26:25]),int(len(QW_Bursts_all_i)/2)))
			btLFP = np.zeros((int(len(QW_Bursts_all_i)/2),len(Win_T_down[1900:2100])))
			# Iterate through all bursts performing spectrogram on 4s of LFP surrounding burst
			# N.B. Burst vectors contain start and end times so we divide the len by 2
			for y in range(0,int(len(QW_Bursts_all_i)/2)):
				x = int(y*2) # To capture every other value in BURSTS vector (i.e. the start times)
				if (((QW_Bursts_all_i[x]-int(2/dt)) >= 0) and ((QW_Bursts_all_i[x]+int(2/dt)) <= int(Time[-1]/dt))):
					'''
					# Define the window as +-2s from the start of the burst
					Win_LFP = LFP[(QW_Bursts_all_i[x]-int(2/dt)):(QW_Bursts_all_i[x]+int(2/dt))]
					Win_LFP = Win_LFP-np.mean(Win_LFP)
					Win_LFP_smooth = signal.convolve(Win_LFP[0:-1:25],smoothWindow,mode = 'same')
					# Perform the spectrogram function to get the power/frequency matrix for this burst
					#Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					Spect = spectrogram(Win_LFP_smooth,Win_T[0:-1:25])
					# There are several outputs for spectrogram, we only want the first Spect[0]
					TF = Spect[0]
					# Normalise power for each frequency band
					for m in range(1,num_frex):
					#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
						TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
					# Assign this power/freq matrix (TF) to the 3D matrix Burst_specta 
					Burst_spectra[:,:,y] = TF
					#print(np.shape(Win_LFP[0:-1:25]),' vs ',np.shape(smoothWindow))
					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					btLFP[y,:] = Win_LFP_smooth[:-1]
					'''
					##############################################################
					Win_LFP = LFP[(QW_Bursts_all_i[x]-int(2/dt)):(QW_Bursts_all_i[x]+int(2/dt))]
					
					Win_LFP = Win_LFP[0:-1:25]
					Win_LFP = Win_LFP-np.mean(Win_LFP)
					# plt.plot(Win_T[0:-1:25],Win_LFP)
					
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					print('Start = ',Win_LFP[1995],', End = ',Win_LFP[2005])

					gap_fill = np.arange(Win_LFP[1995],Win_LFP[2005]+0.01*Win_LFP[2005],((Win_LFP[2005]+0.01*Win_LFP[2005])-Win_LFP[1995])/10)
					Win_LFP[1995:2005] = gap_fill
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					print(len(CellNames)-i,4,int(len(QW_Bursts_all_i))-1-y)
					Win_LFP_smooth = signal.convolve(Win_LFP,smoothWindow,mode = 'same')
					# plt.plot(Win_T[0:-1:25],Win_LFP_smooth)
					# plt.show()
					
									#Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					Spect = spectrogram(Win_LFP_smooth,Win_T[0:-1:25])
					TF = Spect[0]
					for m in range(1,num_frex-1):
					#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
						TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
					Burst_spectra[:,:,y] = TF
					Win_LFP_test = Win_LFP_smooth
					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					btLFP[y,:] = Win_LFP_smooth[:-1]
					print('stLFP, ',btLFP)
					#print('Win_LFP shape = ',np.shape(Win_LFP_smooth))
					# Plot the spectrogram for each spike
					'''
					fig, axs = plt.subplots(2, 1, sharex=True)
					cs = axs[0].contourf(Win_T[0:-26:25],frex,TF)
					axs[1].plot(Win_T[0:-1:25],Win_LFP_test)
					plt.show()
					'''

					#Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					'''
					Spect = spectrogram(Win_LFP_smooth,Win_T[0:-1:25])
					TF = Spect[0]
					for m in range(1,num_frex-1):
					#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
						TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
					Spike_spectra[:,:,y] = TF
					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					stLFP[y,:] = Win_LFP_smooth[:-1]
					'''
					##############################################################

					# Plot the spectrogram for each burst
					'''
					fig, axs = plt.subplots(2, 1, sharex=True)
					cs = axs[0].contourf(Win_T[0:-26:25],frex,TF)
					axs[1].plot(Win_T[0:-1:25],Win_LFP[0:-1:25])
					plt.show()
					'''


			# Define the 3D matrix to contain the spectrograms for each spike (Freqs x Time x Spikes) 
			#Spike_spectra_ctrl = np.zeros((num_frex,len(Win_T[0:-26:25]),int(len(QW_Bursts_ctrl_all_i))))
			btLFP_ctrl = np.zeros((int(len(QW_Bursts_ctrl_all_i)),len(Win_T_Down[1900:2100])))
			# Iterate through all spikes performing spectrogram on 4s of LFP surrounding spike
			# N.B. Spike vectors contain start and end times so we divide the len by 2
			for y in range(0,int(len(QW_Bursts_ctrl_all_i))):
				x = y #int(y*2) # To capture every other value in BURSTS vector (i.e. the start times)
				if (((QW_Bursts_ctrl_all_i[x]-int(2/dt)) >= 0) and ((QW_Bursts_ctrl_all_i[x]+int(2/dt)) <= int(Time[-1]/dt))):
					# Define the window as +-2s from the start of the spike
					Win_LFP = LFP[(QW_Bursts_ctrl_all_i[x]-int(2/dt)):(QW_Bursts_ctrl_all_i[x]+int(2/dt))]
					Win_LFP = Win_LFP[0:-1:25]
					Win_LFP = Win_LFP-np.mean(Win_LFP)
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					gap_fill = np.arange(Win_LFP[1995],Win_LFP[2005]+0.01*Win_LFP[2005],((Win_LFP[2005]+0.01*Win_LFP[2005])-Win_LFP[1995])/10)
					Win_LFP[1995:2005] = gap_fill
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					print(len(CellNames)-i,4,int(len(QW_Bursts_ctrl_all_i))-1-y)
					Win_LFP_smooth = signal.convolve(Win_LFP,smoothWindow,mode = 'same')
					#plt.plot(Win_T[0:-1:25],Win_LFP_smooth)
					#plt.show()
					'''
					# Perform the spectrogram function to get the power/frequency matrix for this spike
					#Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					# There are several outputs for spectrogram, we only want the first Spect[0]
					TF = Spect[0]
					# Normalise power for each frequency band
					for m in range(1,num_frex):
					#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
						TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
					# Assign this power/freq matrix (TF) to the 3D matrix Spike_specta 
					Spike_spectra_ctrl[:,:,y] = TF
					'''
					#print(np.shape(Win_LFP[0:-1:25]),' vs ',np.shape(smoothWindow))
					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					btLFP_ctrl[y,:] = Win_LFP_smooth[:-1]	


			QW_mean_burst_spect = np.mean(Burst_spectra,axis=2)
			QW_mean_btLFP = np.mean(btLFP,axis=0)
			#print('Mean QW stLFP =',QW_mean_stLFP)
			QW_btLFP_AUC = np.trapz(np.abs(QW_mean_btLFP))
			#print('QW_stLFP_AUC =',np.shape(QW_mean_stLFP))
			QW_btLFP_AUC_all = np.append(QW_btLFP_AUC_all,QW_btLFP_AUC)
			#print('QW_stLFP_AUC_all =',QW_stLFP_AUC_all)
			
			#QW_mean_spike_ctrl_spect = np.mean(Spike_spectra_ctrl,axis=2)
			QW_mean_btLFP_ctrl = np.mean(btLFP_ctrl,axis=0)
			QW_btLFP_AUC_ctrl = np.trapz(np.abs(QW_mean_btLFP_ctrl))
			QW_btLFP_AUC_ctrl_all = np.append(QW_btLFP_AUC_ctrl_all,QW_btLFP_AUC_ctrl)
			# CODE FOR PLOTTING AND SAVING MEAN BURST TRIGGERED SPECTROGRAM
			'''
			fig, axs = plt.subplots(2,1)
			axs[0].contourf(Win_T_down[1000:3000],frex,QW_mean_spike_spect[:,1000:3000])
			axs[1].plot(Win_T_down[1000:3000],QW_mean_stLFP[1000:3000])
			Fig_title = (', Spikes = {}, AUC = {}'.format(len(QW_Spikes_all_i),QW_stLFP_AUC))
			plt.suptitle(Filename + Fig_title)
			#FigName = ("C:/Users/dkapl/Dropbox/Matthew Larkum/Matlab backup 30052017/MATLAB/Sleep project/SQL data/Machine learning/{}/QW/{}_QW_BT_spectrum.png".format(Cluster,Filename))
			#plt.savefig(FigName, dpi=150)
			plt.show()
			'''
			
			if len(np.argwhere(np.isnan(QW_mean_burst_spect))) < 1000:
				Summary_Matrix_QW = np.dstack((Summary_Matrix_QW,QW_mean_burst_spect))
			print('Summary QW mega mat shape =',np.shape(Summary_Matrix_QW))

			#if len(np.argwhere(np.isnan(QW_mean_spike_ctrl_spect))) < 1000:
			#	Summary_Matrix_QW_ctrl = np.dstack((Summary_Matrix_QW_ctrl,QW_mean_spike_ctrl_spect))
			#print('Summary QW mega mat shape =',np.shape(Summary_Matrix_QW_ctrl))		

			'''
			Iterate through NREM episodes and find BURSTS from each state
			-----------------------------------------------------------
			'''
			# Find all bursts during NREM and add them to the vector NREM_Bursts_all
			NREM_Bursts_all = []
			NREM_Bursts_ctrl_all = []
			for j in range(0,len(NREMStartT)):
				NREM_Bursts = Bursts[np.where(np.logical_and(Bursts>=NREMStartT[j], Bursts<=NREMEndT[j]))]
				# Declare a vector of zeros to load with the control timesteps
				NREM_Bursts_ctrl = np.zeros(len(NREM_Bursts),)
				# Fill that vector with random timepoints during this NREM window
				for k in range(0,len(NREM_Bursts)-1):
					NREM_Bursts_ctrl[k] = rnd.uniform(NREMStartT[j],NREMEndT[j])
				# Add the burst times from this NREM episode to the array of bursts for the entire recoring 
				NREM_Bursts_all.extend(NREM_Bursts)
				# Do the same for the control timestamps
				NREM_Bursts_ctrl_all.extend(NREM_Bursts_ctrl)
			print('NREM_Bursts size: ',np.shape(NREM_Bursts_all),np.shape(NREM_Bursts_ctrl_all))
			print('NREM_Bursts: ',NREM_Bursts_all[-1],NREM_Bursts_ctrl_all[-1])
			# Conver burst times to burst indexes
			NREM_Bursts_all_i = [int(x/dt) for x in NREM_Bursts_all]
			NREM_Bursts_ctrl_all_i = [int(x/dt) for x in NREM_Bursts_ctrl_all]
			
			# Define the 3D matrix to contain the spectrograms for each burst (Freqs x Time x Bursts) 
			Burst_spectra = np.zeros((num_frex,len(Win_T[0:-26:25]),int(len(NREM_Bursts_all_i)/2)))
			btLFP = np.zeros((int(len(NREM_Bursts_all_i)/2),len(Win_T_down[1900:2100])))
			# Iterate through all bursts performing spectrogram on 4s of LFP surrounding burst
			# N.B. Burst vectors contain start and end times so we divide the len by 2
			for y in range(0,int(len(NREM_Bursts_all_i)/2)):
				x = int(y*2) # To capture every other value in BURSTS vector (i.e. the start times)
				if (((NREM_Bursts_all_i[x]-int(2/dt)) >= 0) and ((NREM_Bursts_all_i[x]+int(2/dt)) <= int(Time[-1]/dt))):
					'''
					# Define the window as +-2s from the start of the burst
					Win_LFP = LFP[(NREM_Bursts_all_i[x]-int(2/dt)):(NREM_Bursts_all_i[x]+int(2/dt))]
					Win_LFP = Win_LFP-np.mean(Win_LFP)
					Win_LFP_smooth = signal.convolve(Win_LFP[0:-1:25],smoothWindow,mode = 'same')
					# Perform the spectrogram function to get the power/frequency matrix for this burst
					#Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					Spect = spectrogram(Win_LFP_smooth,Win_T[0:-1:25])
					# There are several outputs for spectrogram, we only want the first Spect[0]
					TF = Spect[0]
					# Normalise power for each frequency band
					for m in range(1,num_frex):
					#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
						TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
					# Assign this power/freq matrix (TF) to the 3D matrix Burst_specta 
					Burst_spectra[:,:,y] = TF
					#print(np.shape(Win_LFP[0:-1:25]),' vs ',np.shape(smoothWindow))
					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					btLFP[y,:] = Win_LFP_smooth[:-1]
					'''
					##############################################################
					Win_LFP = LFP[(NREM_Bursts_all_i[x]-int(2/dt)):(NREM_Bursts_all_i[x]+int(2/dt))]
					
					Win_LFP = Win_LFP[0:-1:25]
					Win_LFP = Win_LFP-np.mean(Win_LFP)
					# plt.plot(Win_T[0:-1:25],Win_LFP)
					
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					print('Start = ',Win_LFP[1995],', End = ',Win_LFP[2005])

					gap_fill = np.arange(Win_LFP[1995],Win_LFP[2005]+0.01*Win_LFP[2005],((Win_LFP[2005]+0.01*Win_LFP[2005])-Win_LFP[1995])/10)
					Win_LFP[1995:2005] = gap_fill
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					print(len(CellNames)-i,4,int(len(NREM_Bursts_all_i))-1-y)
					Win_LFP_smooth = signal.convolve(Win_LFP,smoothWindow,mode = 'same')
					# plt.plot(Win_T[0:-1:25],Win_LFP_smooth)
					# plt.show()
					
									#Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					Spect = spectrogram(Win_LFP_smooth,Win_T[0:-1:25])
					TF = Spect[0]
					for m in range(1,num_frex-1):
					#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
						TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
					Burst_spectra[:,:,y] = TF
					Win_LFP_test = Win_LFP_smooth
					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					btLFP[y,:] = Win_LFP_smooth[:-1]
					print('stLFP, ',btLFP)
					#print('Win_LFP shape = ',np.shape(Win_LFP_smooth))
					# Plot the spectrogram for each spike
					'''
					fig, axs = plt.subplots(2, 1, sharex=True)
					cs = axs[0].contourf(Win_T[0:-26:25],frex,TF)
					axs[1].plot(Win_T[0:-1:25],Win_LFP_test)
					plt.show()
					'''

					#Spect = spectrogram(Win_LFP[0:-1:25],Win_T[0:-1:25])
					'''
					Spect = spectrogram(Win_LFP_smooth,Win_T[0:-1:25])
					TF = Spect[0]
					for m in range(1,num_frex-1):
					#	tf[i,:] = 10*np.log10(np.divide(tf[i,:],np.mean(tf[i,:])))
						TF[m,:] = 100 * np.divide(np.subtract(TF[m,:],np.mean(TF[m,:])), np.mean(TF[m,:]))
					Spike_spectra[:,:,y] = TF
					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					stLFP[y,:] = Win_LFP_smooth[:-1]
					'''
					##############################################################

					# Plot the spectrogram for each burst
					'''
					fig, axs = plt.subplots(2, 1, sharex=True)
					cs = axs[0].contourf(Win_T[0:-26:25],frex,TF)
					axs[1].plot(Win_T[0:-1:25],Win_LFP[0:-1:25])
					plt.show()
					'''


			# Define the 3D matrix to contain the spectrograms for each spike (Freqs x Time x Spikes) 
			#Spike_spectra_ctrl = np.zeros((num_frex,len(Win_T[0:-26:25]),int(len(NREM_Bursts_ctrl_all_i))))
			btLFP_ctrl = np.zeros((int(len(NREM_Bursts_ctrl_all_i)),len(Win_T_Down[1900:2100])))
			# Iterate through all spikes performing spectrogram on 4s of LFP surrounding spike
			# N.B. Spike vectors contain start and end times so we divide the len by 2
			for y in range(0,int(len(NREM_Bursts_ctrl_all_i))):
				x = y #int(y*2) # To capture every other value in BURSTS vector (i.e. the start times)
				if (((NREM_Bursts_ctrl_all_i[x]-int(2/dt)) >= 0) and ((NREM_Bursts_ctrl_all_i[x]+int(2/dt)) <= int(Time[-1]/dt))):
					# Define the window as +-2s from the start of the spike
					Win_LFP = LFP[(NREM_Bursts_ctrl_all_i[x]-int(2/dt)):(NREM_Bursts_ctrl_all_i[x]+int(2/dt))]
					Win_LFP = Win_LFP[0:-1:25]
					Win_LFP = Win_LFP-np.mean(Win_LFP)
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					gap_fill = np.arange(Win_LFP[1995],Win_LFP[2005]+0.01*Win_LFP[2005],((Win_LFP[2005]+0.01*Win_LFP[2005])-Win_LFP[1995])/10)
					Win_LFP[1995:2005] = gap_fill
					#plt.plot(Win_T[0:-1:25],Win_LFP)
					print(len(CellNames)-i,4,int(len(NREM_Bursts_ctrl_all_i))-1-y)
					Win_LFP_smooth = signal.convolve(Win_LFP,smoothWindow,mode = 'same')
					#plt.plot(Win_T[0:-1:25],Win_LFP_smooth)
					#plt.show()

					Win_LFP_smooth = Win_LFP_smooth[1900:2101]
					btLFP_ctrl[y,:] = Win_LFP_smooth[:-1]	


			NREM_mean_burst_spect = np.mean(Burst_spectra,axis=2)
			NREM_mean_btLFP = np.mean(btLFP,axis=0)
			#print('Mean NREM stLFP =',NREM_mean_stLFP)
			NREM_btLFP_AUC = np.trapz(np.abs(NREM_mean_btLFP))
			#print('NREM_stLFP_AUC =',np.shape(NREM_mean_stLFP))
			NREM_btLFP_AUC_all = np.append(NREM_btLFP_AUC_all,NREM_btLFP_AUC)
			#print('NREM_stLFP_AUC_all =',NREM_stLFP_AUC_all)
			
			#NREM_mean_spike_ctrl_spect = np.mean(Spike_spectra_ctrl,axis=2)
			NREM_mean_btLFP_ctrl = np.mean(btLFP_ctrl,axis=0)
			NREM_btLFP_AUC_ctrl = np.trapz(np.abs(NREM_mean_btLFP_ctrl))
			NREM_btLFP_AUC_ctrl_all = np.append(NREM_btLFP_AUC_ctrl_all,NREM_btLFP_AUC_ctrl)
			# CODE FOR PLOTTING AND SAVING MEAN BURST TRIGGERED SPECTROGRAM
			'''
			fig, axs = plt.subplots(2,1)
			axs[0].contourf(Win_T_down[1000:3000],frex,NREM_mean_spike_spect[:,1000:3000])
			axs[1].plot(Win_T_down[1000:3000],NREM_mean_stLFP[1000:3000])
			Fig_title = (', Spikes = {}, AUC = {}'.format(len(NREM_Spikes_all_i),NREM_stLFP_AUC))
			plt.suptitle(Filename + Fig_title)
			#FigName = ("C:/Users/dkapl/Dropbox/Matthew Larkum/Matlab backup 30052017/MATLAB/Sleep project/SQL data/Machine learning/{}/NREM/{}_NREM_BT_spectrum.png".format(Cluster,Filename))
			#plt.savefig(FigName, dpi=150)
			plt.show()
			'''
			
			if len(np.argwhere(np.isnan(NREM_mean_burst_spect))) < 1000:
				Summary_Matrix_NREM = np.dstack((Summary_Matrix_NREM,NREM_mean_burst_spect))
			print('Summary NREM mega mat shape =',np.shape(Summary_Matrix_NREM))

			#if len(np.argwhere(np.isnan(NREM_mean_spike_ctrl_spect))) < 1000:
			#	Summary_Matrix_NREM_ctrl = np.dstack((Summary_Matrix_NREM_ctrl,NREM_mean_spike_ctrl_spect))
			#print('Summary NREM mega mat shape =',np.shape(Summary_Matrix_NREM_ctrl))		
		except:
			pass







	# Declare files to save arrays to 
	QW_btLFP_AUC_ratio = np.divide(QW_btLFP_AUC_all,QW_btLFP_AUC_ctrl_all)
	NREM_btLFP_AUC_ratio = np.divide(NREM_btLFP_AUC_all,NREM_btLFP_AUC_ctrl_all)

	Summary_NREM = TemporaryFile()
	Summary_QW = TemporaryFile()

	#print('NaNs in NREM = ',np.argwhere(np.isnan(Summary_Matrix_NREM)))
	#print('NaNs in QW = ',np.argwhere(np.isnan(Summary_Matrix_QW)))


	#np.save(Summary_NREM,Summary_Matrix_NREM)
	#np.save(Summary_QW,Summary_Matrix_QW)

	Summary_Matrix_NREM_mean = np.mean(Summary_Matrix_NREM,axis=2)
	Summary_Matrix_QW_mean = np.mean(Summary_Matrix_QW,axis=2)

	#Summary_Matrix_NREM_ctrl_mean = np.mean(Summary_Matrix_NREM_ctrl,axis=2)
	#Summary_Matrix_QW_ctrl_mean = np.mean(Summary_Matrix_QW_ctrl,axis=2)
	
	df = pd.DataFrame({"QW btLFP AUC" : QW_btLFP_AUC_ratio})
	Filename_QW_AUC = ("QW_btLFP_noSpikes_AUC_Cluster_{}.csv".format(Cluster_no))
	df.to_csv(Filename_QW_AUC, index=False)


	df = pd.DataFrame({"NREM btLFP AUC" : NREM_btLFP_AUC_ratio})
	Filename_NREM_AUC = ("NREM_btLFP_noSpikes_AUC_Cluster_{}.csv".format(Cluster_no))
	df.to_csv(Filename_NREM_AUC, index=False)

	df = pd.DataFrame(Summary_Matrix_NREM_mean[:,1000:3000])
	Filename_NREM_meanSpect = ("NREM_btLFP_noSpikes_meanSpect_{}.csv".format(Cluster_no))
	df.to_csv(Filename_NREM_meanSpect, index=False)

	df = pd.DataFrame(Summary_Matrix_QW_mean[:,1000:3000])
	Filename_QW_meanSpect = ("QW_btLFP_noSpikes_meanSpect_{}.csv".format(Cluster_no))
	df.to_csv(Filename_QW_meanSpect, index=False) 

	Filename_NREM_Spect = ("NREM_btLFP_noSpikes_Spect_{}.npy".format(Cluster_no))
	with open(Filename_NREM_Spect,'wb') as f:
		np.save(f,Summary_Matrix_NREM[:,1000:3000,:])
	
	Filename_QW_Spect = ("QW_btLFP_noSpikes_Spect_{}.npy".format(Cluster_no))
	with open(Filename_QW_Spect,'wb') as f:
		np.save(f,Summary_Matrix_QW[:,1000:3000,:])	
	

	#df = pd.DataFrame(Summary_Matrix_NREM_ctrl_mean[:,1000:3000])
	#Filename_NREM_meanSpect_ctrl = ("NREM_stLFP_ctrl_meanSpect_{}.csv".format(Cluster_no))
	#df.to_csv(Filename_NREM_meanSpect_ctrl, index=False)

	#df = pd.DataFrame(Summary_Matrix_QW_ctrl_mean[:,1000:3000])
	#Filename_QW_meanSpect_ctrl = ("QW_stLFP_ctrl_meanSpect_{}.csv".format(Cluster_no))
	#df.to_csv(Filename_QW_meanSpect_ctrl, index=False) 
	'''
	fig, axs = plt.subplots(2, 1, sharex=True)
	cs = axs[0].contourf(Win_T_down[1000:3000],frex,Summary_Matrix_NREM_mean[:,1000:3000])
	axs[0].set(xlabel='Time (s)', ylabel='Frequency (Hz)') 
	axs[0].set_title('NREM')
	cs = axs[1].contourf(Win_T_down[1000:3000],frex,Summary_Matrix_QW_mean[:,1000:3000])
	axs[1].set(xlabel='Time (s)', ylabel='Frequency (Hz)') 
	axs[1].set_title('QW')
	plt.show()
	'''