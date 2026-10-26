"""
This is a simplified Ising sampler that works only with q=2 
and accepts as input J and h directly.
"""

import itertools as it
import SBM.MonteCarlo.MCMC_Potts.MonteCarlo_Potts as mc #type: ignore
import SBM.MonteCarlo.MCMC_PottsProf.MonteCarlo_PottsProf as mcp # type: ignore
import numpy as np #type: ignore
from Bio import SeqIO #type: ignore
from tqdm import tqdm #type: ignore
from scipy.spatial.distance import squareform #type: ignore
import csv as csv

def Create_modAlign(J,h,N,delta_t = None,ITER='',temperature=1):
	"""
	Function to create a alignment based on the provided parameters
	using the C_MonteCarlo module implemented in cython

	Args:
	- output (dict): A dictionary containing 'h' and 'J' values.
	- N (int): The value of N.
	- delta_t (int): Number of MCMC steps
	- ITER (str): Optional parameter (if several 'h' and 'J' values are stored in the output dictionary
	we can choose 'h' and 'J' values at a specific iteration)
    - temperature (float): Optional parameter to specify the temperature at which sampling should proceed

	Returns:
	- numpy.array: A 2D numpy array with the created alignment.
	"""
	L = h.shape[0]
	q=2
	h = np.copy(h)/temperature
	if J is not None: J = np.copy(J)/temperature
	w = np.array(Wj(J,h))
	states = np.random.randint(q,size=(N,L)).astype('int32')
	if J is None: mcp.MC(w,states,int(delta_t),int(q))
	else: mc.MC(w,states,int(delta_t),int(q))
	MSA = np.copy(states)
	return np.array(MSA,dtype = 'int64')

def Wj(J=None, h=None):
	"""
	Build the W vector expected by the C-MonteCarlo routines (mc.MC / mcp.MC),
	for q=2, starting from plain Ising-style J (L,L) and h (L,) using the
	{0,1} convention (states a,b used directly as 0/1 multipliers, not mapped
	to +-1 spins):
	  - pairwise energy term for states (a,b) at sites (i,j): J[i,j] * a * b
	  - field term for state a at site i:                    h[i]   * a

	J is symmetrized (J <- (J + J.T)/2) before use, in case of small
	asymmetries from the fitting procedure.
	"""
	q = 2

	if J is not None:
		J = (J + J.T) / 2  # enforce symmetry
		L = J.shape[0]
		if L > 1:
			W = np.zeros(int(q*L + q*q*L*(L-1)/2))
			x = np.array([[i, j] for i, j in it.combinations(range(L), 2)])
			for a in range(q):
				for b in range(q):
					idx = (q**2*((x[:,0])*(2*L-x[:,0]-1)/2 + (x[:,1]-x[:,0]-1)) + a*q + b).astype(int)
					W[idx] = J[x[:,0], x[:,1]] * a * b
			x = np.arange(L)
			for a in range(q):
				W[(q**2*L*(L-1)/2 + q*x[:] + a).astype(int)] = h[x[:]] * a
	else:
		L = h.shape[0]
		W = np.zeros(q*L)
		for a in range(q):
			W[q*np.arange(L) + a] = h * a
	return W

if __name__ == "__main__":
	J = np.load('Exact_min_J.npy')
	h = np.load('Exact_min_h.npy')
	N = 100000
	X=Create_modAlign(J,h,N,delta_t = 100000 ,ITER='',temperature=1)
	np.save('data/ConformationsArt/MM2.npy',X)