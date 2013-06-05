#!/usr/bin/env python2.7
import matplotlib.pyplot as plt
import numpy as np

testfile = 'spam.test'
predfile = 'spam.predictions'

def scan(name, function): 
  file = open(name, 'r')
  L = []
  while 1:
    line = file.readline()
    if not line: break
    L.append(function(line))
  file.close()
  return L

def process(line):
  if line[0] == '-':
    return 0
  else:
    return 1

def compare(testset, predset):
  # P = true positive
  # N = true negative
  # p = false positive
  # n = false negative
  if (len(testset) != len(predset)):
    print "mismatched file length"
    return
  i  = 0
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for value in testset:
    if value > 0:
      if (value == predset[i]):
        # True Positive
        TP += 1
      else:
        # False Negative
        FP += 1
    else:
      if (value == predset[i]):
        # True Negative
        TN += 1
      else:
        # False Positive
        FN += 1
    i += 1
  TP = float(TP)
  FP = float(FP)
  TN = float(TN)
  FN = float(FN)
  P  = float(TP + FP)
  N  = float(TN + FN)

  Recall    = (float(TP) / float(P))
  Precision = (float(TP) / (float(TP)+float(FP)))
  Accuracy  = (((float(TP) + float(TN))) / (float(P) + float(N)))
  FPR = float(FP) / (float(TN) + float(FP))
  TPR = float(TP) / (float(TP) + float(FN))
  return [TP,FP,TN,FN,P,N,FPR,TPR,Precision,Recall,Accuracy,len(testset)]

testset = scan(testfile, process)
predset = scan(predfile, process)
result0 = compare(testset, predset)

TP        = result0[0]
FP        = result0[1]
TN        = result0[2]
FN        = result0[3]
P         = result0[4]
N         = result0[5]
FPR       = result0[6]
TPR       = result0[7]
Precision = result0[8]
Recall    = result0[9]
Accuracy  = result0[10]
length    = result0[11]

print
print "TP: %d" % TP
print "FP: %d" % FP
print "TN: %d" % TN
print "FN: %d" % FN
print "P:  %d" % P
print "N:  %d" % N
print
print "FPR : %.2f" % FPR
print "TPR : %.2f" % TPR
print
print "Precision : %.2f" % Precision
print "Recall    : %.2f" % Recall
print "Accuracy  : %.2f" % Accuracy
#print "F-Measure : %s" % FMeasure

threshold = []
result    = []
plot      = []
subTest   = []
subPred   = []

for i in range(0, 20):
  threshold.append( int((length / 20)*(i + 1)))

for i in range(0, 20):
  subTest.append(testset[:threshold[i]])
  subPred.append(predset[:threshold[i]])

for i in range(0, 20):
  r = compare(subTest[i], subPred[i])
  result.append(r)

for i in range(0, 20):
  plot.append([result[i][6], result[i][7]])

fig = plt.figure(1, figsize=(5.25, 5))
ax  = fig.add_subplot(111)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
fig.subplots_adjust(top=0.95)
fig.subplots_adjust(bottom=0.14)
fig.subplots_adjust(left=0.18)
fig.subplots_adjust(right=0.95)

plt.plot(0,0)
for p in plot:
  plt.plot(p[0], p[1], 'ro')
plt.plot(1,1)
plt.savefig('ROC.png', dpi=150)
