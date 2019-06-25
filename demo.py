import pickle

import numpy as np
np.seterr(divide='ignore')  # these warnings are usually harmless for this code

from tsc.tsc import TransitionStateClustering
from evaluation.Evaluator import *
from evaluation.Metrics import *

# creates a system whose regimes are uniformly sampled from the stochastic params
sys = createNewDemonstrationSystem(k=3, dims=2, observation=[0.1, 0.1], resonance=[0.0, 0.0], drift=[0.0, 0.0])
t = sampleDemonstrationFromSystem(sys, np.ones((2, 1)), lm=0.0, dp=0.0)
t2 = sampleDemonstrationFromSystem(sys, np.ones((2, 1)), lm=0.0, dp=0.0)
t3 = sampleDemonstrationFromSystem(sys, np.ones((2, 1)), lm=0.0, dp=0.0)

a = TransitionStateClustering(window_size=3, normalize=False, pruning=0.3, delta=-1)

data = pickle.load(open('cached_pos.data', 'rb'))
# a.addDemonstration(np.array(t[0]))
a.addDemonstration(data)
# a.addDemonstration(np.array(t2[0]))
# a.addDemonstration(np.array(t3[0]))
a.fit()

print(a.segmentation)
