import numpy as np

def test_ergodic(transitionmat):
	#print("Hello")
	Mergodic = transitionmat

	# eigenvalues
	eigenvalues, eigenvectors = np.linalg.eig(Mergodic.T)
	print("Eigenvalue approach, eigenvalues: ", eigenvalues)
	# get the corresponding eigenvector as the stationary state vector
	stationarystate= eigenvectors[:,np.where(eigenvalues==max(eigenvalues))] # 1 will always be the highest
	stationarystatenorm = np.squeeze(np.asarray(stationarystate))/sum(stationarystate).item(0)
	print("stationary state from eigenvector = " + str(stationarystatenorm))
	err_tol = 1e-9 # check to make sure this eigenvalue is close to 1
	if (abs(max(eigenvalues)-1)> err_tol):
	    print("highest eigenvalue not close to 1"+str(max(eigenvalues)))


	# matrix multiplication
	Minfty = (Mergodic.T)**64
	print("from matrix multiplication, stationary state= " +str(Minfty[:,1]))

	# start sims
	numstates = transitionmat.shape[0]
	# choose a distribution for initial state
	initdist= (list(range(numstates))+np.ones(numstates))/(sum(list(range(numstates)))+numstates)
	np.random.seed(100)

	numexp = 100
	numiter = 1000
	# matrix to store the state at each iteration of the markov chain
	statemat = np.zeros((numexp, numiter))
	startstates = np.zeros(numexp)

	for exp in range(numexp):
	    # choose start state for this experiment:
		currstate= np.random.choice(a= list(range(numstates)), p=np.squeeze(np.asarray(initdist)))+1
		startstates[exp] = currstate
		# iterate around the markov chain 100 times
		for niter in range(numiter):
			statevec = np.zeros((numstates,1))
			statevec[currstate-1]=1
			statetransprobs = np.dot(Mergodic.T, statevec)
			currstate = np.random.choice(a= list(range(numstates)), p=np.squeeze(np.asarray(statetransprobs)))+1
			statemat[exp, niter]= currstate

	unique, counts = np.unique(statemat, return_counts=True)
	print("stationary state from simulatons: " + str(counts/np.product(statemat.shape)))

#testmat= np.matrix([[0.3, 0.3, 0.2, 0.2], [0.4, 0.3, 0.2, 0.1], [0.3, 0.1, 0.2, 0.4], [0.1, 0.1, 0.1, 0.7]])
#test_ergodic(testmat)
