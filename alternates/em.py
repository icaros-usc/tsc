import numpy as np
from sklearn import mixture, decomposition
from dpcluster import *
import matplotlib.pyplot as plt
import dtw
import autoregressive.distributions as di

"""
Forward-backward algorithm 
"""
class EMForwardBackward:

	def __init__(self, n_components, n_iter = 20, pruning=0.8, verbose=True):
		self.verbose = verbose
		self.model = []
		self.task_segmentation = []
		self.segmentation = []
		
		#internal variables not for outside reference
		self._demonstrations = []
		self._demonstration_sizes = []
		self._transitions = []
		self._transition_states_scluster = []

		self.n_components = n_components
		self.n_iter = n_iter
		self.pruning = pruning

	def addDemonstration(self,demonstration):
		demonstration = np.squeeze(np.array(demonstration))
		demo_size = np.shape(demonstration)
		
		if self.verbose:
			print "[EM] Adding a Demonstration of Size=", demo_size

		self._demonstration_sizes.append(demo_size)

		self._demonstrations.append(demonstration)



	def fit(self):
		data = np.squeeze(np.array(self._demonstrations[0])) 
		obs_dim = data.shape[1]

		#initializing A matrices randomly
		transition_matrices = []
		for i in range(0, self.n_components):
			transition_matrices.append(np.random.randn(obs_dim,obs_dim))


		#initializing covariance matrices as identity
		covariance_matrices = []
		for i in range(0, self.n_components):
			covariance_matrices.append(np.eye(obs_dim))

		#initializing likelihoods
		lp = []

		#actual em iterations
		for iteration in range(0, self.n_iter):
			
			#calculate likelihoods and re-normalize

			lpp = []
			for i in range(0, len(self._demonstrations)):

				demoassignment = []
				for j in range(1, len(self._demonstrations[i])):
					
					newp = np.zeros((self.n_components,1))

					for k in range(0, self.n_components):
						xt = np.matrix(self._demonstrations[i][j-1])
						xtt = np.matrix(self._demonstrations[i][j])
						res = xtt.T - transition_matrices[k]*xt.T
						ll = np.dot(np.dot(res.T, np.linalg.inv(covariance_matrices[k])), res)
						scaling = 1.0/np.sqrt(np.linalg.det(covariance_matrices[k])*(2*np.pi)**obs_dim)

						newp[k] = scaling*np.exp(-ll) 


					newp = newp / np.sum(newp)

					#print newp


					demoassignment.append(newp)

				lpp.append(demoassignment)

			#print lpp, len(lpp[0])

			#calculate transition and covariance matrices
			new_transition_matrices = []
			new_covariance_matrices = []
			for k in range(0, self.n_components):

				outer_product = np.zeros((2,2))
				cross_product = np.zeros((2,2))

				#transitions
				for i in range(0, len(self._demonstrations)):
					for j in range(1, len(self._demonstrations[i])):
						xt = np.matrix(self._demonstrations[i][j-1])
						xtt = np.matrix(self._demonstrations[i][j])
						#print len(lpp), i, len(lpp[i]), j
						#print i,j, lpp[i][j-1][k]
						outer_product = outer_product + np.squeeze(lpp[i][j-1][k]) * (xt.T * xt)
						cross_product = cross_product + np.squeeze(lpp[i][j-1][k]) * (xt.T * xtt)

				#for numerical instability
				A = (np.linalg.inv(outer_product)*cross_product)

				new_transition_matrices.append(A)


				outer_product = np.zeros((2,2))
				normalization = 0

				#covariances
				for i in range(0, len(self._demonstrations)):
					for j in range(1, len(self._demonstrations[i])):
						xt = np.matrix(self._demonstrations[i][j-1])
						xtt = np.matrix(self._demonstrations[i][j])
						res = xtt.T - np.dot(A, xt.T)
						outer_product = outer_product + np.squeeze(lpp[i][j-1][k])*(res * res.T)
						normalization = np.squeeze(lpp[i][j-1][k]) + normalization

				
				new_covariance_matrices.append(outer_product/normalization)

			print "[EM] Iteration", iteration
			transition_matrices  = new_transition_matrices
			covariance_matrices = new_covariance_matrices
			lp = lpp


		#print len(lp), len(lp[0])
		self.findTransitions(lp)
		self.clusterInState()
		self.pruneClusters()
		self.clusterInTime()
		self.taskToTrajectory()


	def findTransitions(self, lp):
		transitions = []

		for i in range(0, len(self._demonstrations)):

				assignment_index = []

				for j in range(1, len(self._demonstrations[i])):
					vp = np.argmax(lp[i][j-1])
					assignment_index.append(vp)

				assignment_index = self.smoothing(assignment_index)
					
				for j in range(1, len(assignment_index)):
					if assignment_index[j] != assignment_index[j-1]:
						transitions.append((i,j))

		self._transitions = transitions

		return self._transitions






	"""
		@@Taken from TSC code
	"""


	"""
	This applies smoothing to the indices to make sure
	rapid changes are discouraged
	"""
	def smoothing(self, indices):
		newIndices = indices
		for i in range(1,len(indices)-1):
			if indices[i] != indices[i-1] and indices[i] != indices[i+1] and indices[i+1] == indices[i-1]:
			   newIndices[i] = indices[i+1]

			   if self.verbose:
			   	print "[EM] Smoothed out index=",i

		return newIndices

	
	"""
	This prunes transitions to a specified threshold
	"""
	def pruneClusters(self):
		distinct_clusters = set([c[2] for c in self._transition_states_scluster])
		N = len(self._demonstration_sizes)
		new_transitions = []
		for c in distinct_clusters:
			tD = set([d[0] for d in self._transition_states_scluster if d[2] == c])
			tS = [d for d in self._transition_states_scluster if d[2] == c]
			if (len(tD) +0.0)/N > self.pruning:
				new_transitions.extend(tS)

		if self.verbose:
			print "[TSC] Transitions Before Pruning=", self._transition_states_scluster, "After=",new_transitions

		self._transition_states_scluster = new_transitions



	"""
	Takes the task segmentation and returns a trajectory
	segmentation. For conditioning reasons this doesn't 
	use DP-GMM but finds all clusters of size segmentl (automatically set)
	"""
	def taskToTrajectory(self):
		N = len(self._demonstration_sizes)
		for i in range(0,N):
			tSD = [(k[2],k[3],k[1]) for k in self.task_segmentation if k[0] == i]
			
			timeDict = {}
			for t in tSD:
				key = (t[0], t[1])
				if  key in timeDict:
					timeDict[key].append(t[2])
				else:
					timeDict[key] = [t[2]]
			
			print timeDict

			tseg = [np.median(timeDict[k]) for k in timeDict]
			tseg.append(0)
			tseg.append(self._demonstration_sizes[i][0])
			self.segmentation.append(tseg)

	"""
	Runs multiple runs of DPGMM takes the best clustering
	"""
	def DPGMM(self,data, dimensionality, p=0.9, k=1):
		runlist = []
		for i in range(0,k):
			runlist.append(self.DPGMM_Helper(data,dimensionality,p))
		runlist.sort()

		print runlist

		#return best
		return runlist[-1][1]

	"""
	Uses Teodor's code to do DP GMM clustering
	"""
	def DPGMM_Helper(self,data, dimensionality, p=0.9):
		vdp = VDP(GaussianNIW(dimensionality))
		vdp.batch_learn(vdp.distr.sufficient_stats(data))		
		likelihoods = vdp.pseudo_resp(np.ascontiguousarray(data))[0]

		real_clusters = 1
		cluster_s = vdp.cluster_sizes()
		total = np.sum(cluster_s)
		running_total = cluster_s[0]
		for i in range(1,len(vdp.cluster_sizes())):
			running_total = running_total + cluster_s[i]
			real_clusters = i + 1
			if running_total/total > p:
				break

		return (-np.sum(vdp.al), [np.argmax(l[0:real_clusters]) for l in likelihoods])

	"""
	This function applies the state clustering
	"""
	def clusterInState(self):
		tsN = len(self._transitions)
		p = self._demonstration_sizes[0][1]
		ts_data_array = np.zeros((tsN,p))

		for i in range(0, tsN):
			ts = self._transitions[i]
			ts_data_array[i,:] = self._demonstrations[ts[0]][ts[1],:]


		#Apply the DP-GMM to find the state clusters
		indices = self.DPGMM(ts_data_array,p)
		indicesDict = list(set(indices))

		self._transition_states_scluster = []
		self._distinct_state_clusters = 0
		
		if self.verbose:
			print "[EM] Removing previously learned state clusters "

		#encode the first layer of clustering:
		for i in range(0,tsN):
			label = indicesDict.index(indices[i])
			tstuple = (self._transitions[i][0], self._transitions[i][1], label)
			self._transition_states_scluster.append(tstuple)

		self._distinct_state_clusters = len(list(set(indices)))
		#print self._distinct_state_clusters

		if self.verbose:
			print "[EM] Discovered State Clusters (demoid, time, statecluster): ", self._transition_states_scluster

	"""
	This function applies the time sub-clustering
	"""
	def clusterInTime(self):
		p = self._demonstration_sizes[0][1]

		unorderedmodel = []

		for i in range(0,self._distinct_state_clusters):
			tsI = [s for s in self._transition_states_scluster if s[2]==i]
			ts_data_array = np.zeros((len(tsI),p))
			t_data_array = np.zeros((len(tsI),2))
			
			for j in range(0, len(tsI)):
				ts = tsI[j]
				ts_data_array[j,:] = self._demonstrations[ts[0]][ts[1],:]

				t_data_array[j,0] = ts[1] + np.random.randn(1,1) #do this to avoid conditioning problems
				t_data_array[j,1] = ts[1] + np.random.randn(1,1) #do this to avoid conditioning problems

			if len(tsI) == 0:
				continue

			#Since there is only one state-cluster use a GMM
			mm  = mixture.GMM(n_components=1)
			mm.fit(ts_data_array)


			#subcluster in time
			indices = self.DPGMM(t_data_array,2,0.9)
			#print t_data_array, indices
			indicesDict = list(set(indices))

			#finish off by storing two values the task segmentation	
			for j in range(0, len(tsI)):
				dd = set([tsI[n][0] for (n, ind) in enumerate(indices) if ind == indices[j]])
				
				#time pruning condition
				if (len(dd) + 0.0)/len(self._demonstration_sizes) < self.pruning:
					continue

				self.task_segmentation.append((tsI[j][0],
										  	   tsI[j][1],
										       tsI[j][2],
										       indicesDict.index(indices[j])))

			#GMM model
			unorderedmodel.append((np.median(t_data_array),mm))

		unorderedmodel.sort()
		self.model = [u[1] for u in unorderedmodel]

		if self.verbose:
			print "[TSC] Learned The Following Model: ", self.model


	#does the compaction
	def compaction(self,delta=-1):
		for i in range(0, len(self._demonstrations)):
			segs = self.segmentation[i]
			segs.sort()
			d = self._demonstrations[i]

			prev = None
			removal_vals = []

			for j in range(0,len(segs)-1):
				cur = d[segs[j]:segs[j+1],:]

				if prev != None and len(cur) > 0 and len(prev) > 0:
					dist, cost, acc, path = dtw.dtw(cur, prev, dist=lambda x, y: np.linalg.norm(x - y, ord=2))
					cmetric = dist/len(path)
					if cmetric < delta:
						removal_vals.append(segs[j+1]) 

						if self.verbose:
							print "[TSC] Compacting ", segs[j], segs[j+1]

				prev = cur

			self.segmentation[i] = [s for s in self.segmentation[i] if s not in removal_vals]

