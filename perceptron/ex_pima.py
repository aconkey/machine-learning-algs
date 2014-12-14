#!/usr/bin/python

import os
import pylab as pl
import numpy as np
import perceptron as pcn

os.chdir("/home/adam/github/marsland-machine-learning/datasets")
pima = np.loadtxt('pima-indians-diabetes.data',delimiter=',')
inputs = pima[:,:8]
outputs = pima[:,8:9]

p = pcn.perceptron(inputs,outputs)
p.train(inputs,outputs,0.25,100)
p.confmat(inputs,outputs)
