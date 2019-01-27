import numpy as np
from numpy.linalg import inv

def test_absorbic(transitionmat):
	# calculate absorbic properties
	Mabs = transitionmat
	Mtrans= Mabs[1:3, 1:3]
	m4 = Mabs[1:3, 3] # the transition probabilities for state 4
	m1 = Mabs[1:3, 0] # the transition probabilities for state 1

	# absorbing probabilities for state 4:
	f4 = np.dot(inv(np.identity(2)- Mtrans),m4)

	# absorbing probabilities for state 4:
	f1 = np.dot(inv(np.identity(2)- Mtrans),m1)

	fprobs = np.concatenate((f1, f4), axis=1)

	# run markov chain many times to compute distribution
	np.random.seed(77)
	absorbic_states = [1, 4]
	transient_states = [2, 3]
	numexp=10000
	simfprobs = np.zeros((2,2))
	numstates=Mabs.shape[0]

	for transstate in transient_states:
	    finalstates = np.zeros(numexp)
	    for i in range(numexp):
	        currstate=transstate
	        while currstate not in absorbic_states:
	            statevec = np.zeros((numstates,1))
	            statevec[currstate-1]=1
	            statetransprobs = np.dot(Mabs.T, statevec)
	            currstate = np.random.choice(a= list(range(numstates)), p=np.squeeze(np.asarray(statetransprobs)))+1
	        finalstates[i]=currstate
	    unique, counts = np.unique(finalstates, return_counts=True)
	    #finalstatecounts= dict(zip(unique, counts)) 
	    simfprobs[transstate-2,:]=counts/numexp


	print(" from simulation:")
	print(" The final state absorbing probabilities starting from state 2 are, f21: " + str(simfprobs[0,0])+ " and f24: "+str(simfprobs[0,1]))
	print(" The final state absorbing probabilities starting from state 3 are, f31: " + str(simfprobs[1,0])+ " and f34: "+str(simfprobs[1,1]))

	print("\n from exact calculation:")
	print(" The final state absorbing probabilities starting from state 2 are, f21: " + str(fprobs[0,0])+ " and f24: "+str(fprobs[0,1]))
	print(" The final state absorbing probabilities starting from state 3 are, f31: " + str(fprobs[1,0])+ " and f34: "+str(fprobs[1,1]))

	print("\n Difference between simulation and calculation:")
	print(" f21: " + str(simfprobs[0,0]-fprobs[0,0]))
	print(" f24: " + str(simfprobs[0,1]-fprobs[0,1]))
	print(" f31: " + str(simfprobs[1,0]-fprobs[1,0]))
	print(" f34: " + str(simfprobs[1,1]-fprobs[1,1]))

#test_absorbic(np.matrix([[1,0,0,0], [0.2, 0.2, 0.2, 0.4], [0.25, 0.25, 0.25, 0.25], [0,0,0,1]]))
