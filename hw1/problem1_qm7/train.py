import os
import pickle
import sys
import numpy as np
import model as nn
import copy
import scipy
import scipy.io 
import argparse
from tqdm import tqdm
from utils import rotate_3d_coordinates
from plots import plot_rmse
from pathlib import Path

def train(nnsgd, nnavg, R, Z, E_true, mb, hist, augment):
	rmses = []
	"""
	R: (N_train, 23, 3)
	Z: (N_train,23)
	E_true: (N_train,)
	"""
	for i in tqdm(range(1,100000)):

		# learning rate schedule
		if i > 0:     lr = 0.0000001  
		if i > 500:   lr = 0.00000025
		if i > 2500:  lr = 0.0000005
		# if i > 12500: lr = 0.000001

		

		#sample minibatch indices
		r = np.random.randint(0,len(R),[mb]) #generate an array of length mb (batch size) with indices randomly sampled from 0 to len(R) - 1
		#--> creates random mini batch of size mb --> (mb, 23, 3)

		Rs = R[r]
		Zs = Z[r]

		if augment:
			Rs_rotated = []
			x_deg = np.random.randint(-180, 180)
			y_deg = np.random.randint(-180, 180)
			z_deg = np.random.randint(-180, 180)

			for molecule in Rs: #each molecule in minibatch rotated independently
				molecule_rotated = rotate_3d_coordinates(molecule, x_deg, y_deg, z_deg)
				Rs_rotated.append(molecule_rotated)

			Rs = np.stack(Rs_rotated, axis = 0) #rowise

		inputs = (Rs, Zs)
		targets = E_true[r]
		
		E_pred = nnsgd.forward(inputs)
		loss = 0.5 * np.sum(np.square(E_pred - targets))
		rmse = np.square(E_pred-E_true[r]).mean(axis=0)**.5
		rmses.append(rmse)

		DY = E_pred - targets
		nnsgd.backward(DY) 
		nnsgd.update(lr)
		nnavg.average(nnsgd,(1/hist)/((1/hist)+i))
		nnavg.nbiter = i

		if i % 100 == 0: 
			print(f"LOSS: {loss:5f}")
			print(f"RMSE: {sum(rmses[-100:])/100} kcal/mol")
			pickle.dump(nnavg,open(f'nn-augment={augment}.pkl','wb'),pickle.HIGHEST_PROTOCOL) #save model
	
	np.save(f'training-rmses-augment={augment}.npy', np.array(rmses))
	print("Done Training")
	title = f"RMSE_WITHOUT_3D_augmentation" if not augment else f"RMSE_WITH 3D_augmentation"
	plot_rmse(rmses, title, Path("figures"))


if __name__ == '__main__':
	'''Setup'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=123, help='Random seed')
	parser.add_argument('--augmentation', action='store_true', help='whether to perform data augmentations')
	parser.add_argument('--train_fraction', type=float, default=0.5, help='Fraction of training split to use')
	parser.add_argument('--mb', type=int, default=25, help='Batch size')
	parser.add_argument('--hist', type=float, default=0.1, help='Fraction of the history to be remembered for Exponential Moving Average')
	args = parser.parse_args()

	np.random.seed(args.seed)

	'''Load data'''
	if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
	dataset = scipy.io.loadmat('qm7.mat')
	split_idx = dataset['P'][1:].flatten() #leave first split for testing
	skip_every = int(1/args.train_fraction)
	R = dataset['R'][split_idx][::skip_every]
	Z = dataset['Z'][split_idx][::skip_every]
	E_true = dataset['T'][0,split_idx][::skip_every]

	'''Create neural network'''
	#Define input and output layers
	
	I,O = nn.Input((R, Z)),nn.Output(E_true)
	fcl1 = nn.Linear(I.nbout, 400) #(N*4, 400)
	act1 = nn.Tanh()
	fcl2 = nn.Linear(400, 100)
	act2 = nn.Tanh()
	fcl3 = nn.Linear(100, O.nbinp) #(100, 1)

	modules = [I, fcl1, act1, fcl2, act2, fcl3, O]
	nnsgd = nn.Sequential(modules) 
	nnavg = copy.deepcopy(nnsgd)

	'''Train neural network'''
	train(nnsgd, nnavg, R, Z, E_true, args.mb, args.hist, args.augmentation)
