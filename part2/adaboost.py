#!/usr/bin/env python2.7

import pdb

import random
import numpy as np

from sys import exit
from subprocess import call

# Define original dataset S
trainfile = 'spam.train'
predfile  = 'spam.predictions'
testfile  = 'spam.test'
K = 20 # Boosting iterations
T = 10 # Number of subsets used for training

### Execute a function over every line in file
def scan(name, function): 
  file = open(name, 'r')
  L = []
  while 1:
    line = file.readline()
    if not line: break
    L.append(function(line))
  file.close()
  return L

### Test for sign in a file line
def process(line):
  if line[0] == '-':
    return 0
  else:
    return 1

### Display the difference between two sets
def ldiff(a, b):
  n = 0
  for i in range(0,len(a)):
    if a[i] != b[i]:
      n += 1
  return n

### S = { x_1 ... x_n }
def generate_S():
  # Create the predictions file for the original test set
  status = call("./svm_learn -t 0 spam.train spam.model", shell=True)
  if status != 0:
    print "Unable to run svm_learn"
    exit(-1)
  status = call("./svm_classify spam.test spam.model spam.predictions", shell=True)
  if status != 0:
    print "Unable to run svm_classify"
    exit(-1)
  # Read the predictions file
  file = open(trainfile, 'r')
  L = []
  while 1:
    line = file.readline()
    if not line: break
    L.append(line)
  file.close()
  # Generate S from predictions file
  S = []
  for line in L:
    if line[0] == '-':
      S.append('-')
    else:
      S.append('+')
  return S

### Generate actual classes for comparison
def generate_testdata():
  return scan(testfile, process)

### Assign random weights w_t
def assign_weights(S):
  w_t = []
  for i in S:
    w_t.append(float(1)/len(S))
  return w_t
  
#### Create t subsets of dataset S
def generate_subset(t, S):
  # Read the training file
  S_train = []
  file = open(trainfile, 'r')
  while 1:
    line = file.readline()
    if not line: break
    S_train.append(line)
  file.close()
  ### Create subset
  ### Write out subset to file  
  s_t = []
  s_t_filename = './subsets/' + str(t) + '.train'
  file = open(s_t_filename, 'w')
  for i in range(0,len(S)):
    c = random.choice( range( 0,len(S) ) )
    s_t.append( S[c] )
    file.write(S_train[c])
  file.close()
  return s_t

### Output a file which is readable by ROC.py
def output_hypothesis(h, filename):
  file = open(filename, 'w')
  for i in range(0,len(h)):
    if h[i] == 0:
      file.write('-\n')
    else:
      file.write('1\n')
  file.close()

### Train t weak hypothesis on S_t
def train(t):
  predfile = 'spam.predictions'
  s_t_file = './subsets/' + str(t) + '.train'
  # Create the models file for the original test set
  status = call('./svm_learn -t 0 ./subsets/' + str(t) + '.train ' + 
                './models/' + str(t) + '.model', 
                shell=True)
  if status != 0:
    print "Unable to run svm_learn"
    exit(-1)
  file = open(predfile, 'r')
  L = []
  while 1:
    line = file.readline()
    if not line: break
    L.append(line)
  file.close()
  # Generate S from predictions file
  prediction_t = scan(s_t_file, process)
  return prediction_t

def run(t):
  p_t_file = './predictions/' + str(t) + '.predictions'
  # Create the models file for the original test set
  status = call('./svm_classify ' + 
                'spam.test ' + 
                './models/' + str(t) + '.model ' + 
                './predictions/' + str(t) + '.predictions', 
                shell=True)
  if status != 0:
    print "Unable to run svm_classify"
    exit(-1)
  # Generate S from predictions file
  hypothesis_t = scan(p_t_file, process)
  return hypothesis_t
  

#  ### Calculate Alpha
#  A_t = 1/2 * ln(1-E_t) / E_t
#  ### Recalculate weights w_t
#  w_t = w_t+1_i * exp( -1 * A_t * y_i * h_t(x_i) / size(w)
#### Calculate Ensemble Hypothesis
#s = 0
#for loop t
#  s += A_t * h_t(x_i)
#H_x = sgn(s)
#
#print H_x

def main():
  ### S = { x_1 ... x_n }
  S = generate_S()
  ### Loop over all hypothesis
  w = [] # weights
  s = [] # data subsets
  p = [] # Prediction
  h = [] # Hypothesis
  r = [] # Actual classes
  E = [] # Initial error
  e = [] # Hypothesis error
  a = [] # Alpha
  Z = [] # Normalizer
  H = [] # Ensemble hypothesis

  ### For each hypothesis
  ### Assign weights
  ### Train h(t)
  ### Run h(t)
  for t in (range(0,T)):
    ### Assign random weights w_t
    w.append( assign_weights(S) )
    ### Create t subsets of dataset S
    s.append( generate_subset(t, S) )
    ### Train t weak hypothesis on S_t
    p.append( train(t) )
    ### Run weak hypothesis t on S
    h.append( run(t) )
  ### Duplicate weights for each hypothesis
  w.append( w[0] )
  for t in (range(0,T)):
    ### Get error of hypothesis t
    w.append( w[0] )
    e.append(0)
    for i in (range(0,len(s))):
      for f in (range(0,len(w))):
        if p[t][i] != h[t][i]:
          y_i = 1
        else:
          y_i = 0
        e[t] += w[t][i] * y_i
    ### Get alpha of hypothesis t
    if e[t] == 0:
      a.append( 0 )
    else:
      a.append( .5 * ( np.log( abs( float(1 - e[t]) / e[t] ) ) ) )
    ### Set normalizers for weight recalculation
    Z.append( 0 )
    for f in (range(0,len(s[t]))):
      Z[t] += ( w[t][f] * np.exp( float( (0 - a[t]) * p[t][f] * h[t][f] ) ) )
    ### Recalculate weights for t
    for i in (range(0,len(s[t]))):
      w[t][i] = ( w[t][i] * np.exp( float( (0 - a[t]) * p[t][i] * h[t][i] ) / Z[t] ) )

  ### Generate Ensemble Hypothesis
  for x in (range(0,len(h[0]))):
    H.append(0)
    for t in (range(0,T)):
      for k in (range(0,K)):
        H[x] += ( a[t] * h[t][x] )
      H[x] = np.sign( H[x] )

  ### Get actual classes for comparison
  r = generate_testdata() 

  ### At this point, 
  ### H == our Ensemble result, 
  ### h == our set of 10 weak hypothesis results 
  ### r == our actual classes

  ### Output files for ROC
  H_filename = 'H.test'
  r_filename = 'r.pred'
  output_hypothesis(H, H_filename)
  output_hypothesis(r, r_filename)

  ### Output stats
  H_correct = 0
  h_correct = []
  for t in range(0,T):
      h_correct.append( 0 )
  for i in (range(0,len(r))):
    if (abs(int(H[i])) == r[i]):
      H_correct += 1
    for t in (range(0,len(h))):
      if (abs(int(h[t][i])) == r[i]):
        h_correct[t] += 1

  print "H correct : %d" % H_correct
  for t in range(0,len(h)):
    print "h[%d] correct : %d" % (t, h_correct[t])

  ### Dataset comparison
  #m0 = ldiff(H, h[0])
  #m1 = ldiff(h[0], h[1])
  #m2 = ldiff(h[0], h[2])
  #m3 = ldiff(h[0], h[3])
  #m4 = ldiff(h[0], h[4])
  #m5 = ldiff(h[0], h[5])
  #m6 = ldiff(h[0], h[6])
  #m7 = ldiff(h[0], h[7])
  #m8 = ldiff(h[0], h[8])
  #m9 = ldiff(h[0], h[9])


if __name__ == "__main__":
    main()
