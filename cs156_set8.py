#!/usr/bin/env python
# coding: utf-8

# # Learning Systems - Homework 8
# ## Name: Alina Zhang
# ## Email: h2hang@caltech.edu

# In[1]:


# Imports
import numpy as np
import random
import sys
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


# ### Question 1: D - a quadratic programming problem with d + 1 variables
# We know that for the original formulation of the hard-margin SVM problem, we want to minimize $\frac{1}{2} {w^T}{w}$ subject to $y_n({w^T}x_n +b) \geq 1$ for $n = 1, 2, \dots N$. Because our variables are ${w}$ and $b$, with the former having $d$ variables and the latter having 1 variable, we end up with a quadratic programming problem with $d + 1$ variables.  
# 

# ### Question 2: A - 0 versus all

# In[2]:


# Get data from files
test = np.loadtxt('features.test')
train = np.loadtxt('features.train')

# Separate x, y for test
x_test = test[:, 1:] # All rows of 2nd and 3rd columns
y_test = test[:, 0] # ALl rows of first column

# Separate x, y for training
x_train = train[:, 1:]
y_train = train[:, 0]

# Get N for both
N_test = y_test.size
N_train = y_train.size

# Find error
def error(clf, x, y):
    return 1 - clf.score(x, y)


# In[3]:


# Constants
C = 0.01
Q = 2

def experiment(classif, c, q):
    clf = svm.SVC(kernel='poly', C=c, degree=q, gamma=1.0, coef0=1.0)
    
    train_bin = np.array([1 if y == classif else -1 for y in y_train])
    test_bin = np.array([1 if y == classif else -1 for y in y_test])
    clf.fit(x_train, train_bin)
    return error(clf, x_test, test_bin), sum(clf.n_support_)

for i in range(0, 9, 2):
    print(f"The E_in of {i} versus all was {experiment(i, C, Q)[0]}.")


# ### Question 3: A - 1 versus all
# 

# In[4]:


for i in range(1, 10, 2):
    print(f"The E_in of {i} versus all was {experiment(i, C, Q)[0]}.")


# ### Question 4: C - 1800

# In[5]:


svs = experiment(0, C, Q)[1] - experiment(1, C, Q)[1]
print(f"The difference between the number of support vectors of these two classifiers is {svs}.")


# ### Question 5: D - Maximum C achieves the lowest $ E_{in} $

# In[6]:


Q = 2
Cs = [.001, .01, .1, 1]

def get_1v5(x, y):
    idx = []
    for i in range(y.size):
        if (y[i] == 1) | (y[i] == 5):
            idx.append(i)
    return (x[idx], y[idx])

def exp_1v5(c, q):
    clf = svm.SVC(kernel='poly', C=c, degree=q, gamma=1.0, coef0=1.0)
    
    x_test1, y_test1 = get_1v5(x_test, y_test)
    x_train1, y_train1 = get_1v5(x_train, y_train)
    clf.fit(x_train1, y_train1)
    
    # Returns E_out, E_in, # Support vectors
    return error(clf, x_test1, y_test1), error(clf, x_train1, y_train1), clf.n_support_

print("C\t| E_out\t  | E_in    | Support vectors")
for c in Cs:
    e_out, e_in, sv = exp_1v5(c, Q)
    if (c == 1):
        print(f"{c}\t| {round(e_out, 5)} | {round(e_in, 5)}  | {sv[0]}")
    else:
        print(f"{c}\t| {round(e_out, 5)} | {round(e_in, 5)} | {sv[0]}")


# ### Question 6: B - When C = 0.001, the number of support vectors is lower at Q = 5

# In[7]:


Cs = [.0001, .001, .01, 1]
print("Q = 2:")
print("C\t| E_out\t  | E_in    | Support vectors")
for c in Cs:
    e_out, e_in, sv = exp_1v5(c, 2)
    if (c == 1):
        print(f"{c}\t| {round(e_out, 5)} | {round(e_in, 5)}  | {sv[0]}")
    else:
        print(f"{c}\t| {round(e_out, 5)} | {round(e_in, 5)} | {sv[0]}")
        
print("\nQ = 5:")
print("C\t| E_out\t  | E_in    | Support vectors")
for c in Cs:
    e_out, e_in, sv = exp_1v5(c, 5)
    if ((c == 1.0) | (c == 0.1)):
        print(f"{c}\t| {round(e_out, 5)} | {round(e_in, 5)}  | {sv[0]}")
    else:
        print(f"{c}\t| {round(e_out, 5)} | {round(e_in, 5)} | {sv[0]}")


# ### Question 7: B - C = 0.001 is selected most often

# In[58]:


Q = 2
CV = 10
RUNS = 100
cs = [.0001, .001, .01, .1, 1]

def cross_val(c_list, q):
    x_train1, y_train1 = get_1v5(x_train, y_train)
    rskf = RepeatedStratifiedKFold(n_splits=CV, n_repeats=RUNS)
    best_cs = [] # List of selected Cs
    for train_idx, val_idx in rskf.split(X_train1v5, Y_train1v5):
        x_t, x_v = x_train1[train_idx], x_train1[val_idx]
        y_t, y_v = y_train1[train_idx], y_train1[val_idx]
        high_score = 0
        best_c = None
        for c in c_list:
            clf = svm.SVC(kernel='poly', C=c, degree=q, gamma=1.0, coef0=1.0)
            clf.fit(x_t, y_t)
            score = clf.score(x_v, y_v)
            if score > high_score:
                high_score = score
                best_c = c
        best_cs.append(best_c)
    # Finds the C that appeared most often
    final_c = max(set(best_cs), key = best_cs.count)
    clf = svm.SVC(kernel='poly', C=final_c, degree=q, gamma=1.0, coef0=1.0)
    scores = cross_val_score(clf, x_train1, y_train1, cv=rskf)
    avg_ev = 1 - scores.mean()
    return final_c, avg_ev

print(f"C = {cross_val(cs, Q)[0]} was selected most often.")


# ### Question 8: C - 0.005

# In[60]:


print(f"The average E_ev over 100 runs is closest to {round(cross_val(cs, Q)[1], 3)}.")


# ### Question 9: E - $ 10 ^ 6 $

# In[33]:


cs = [0.01, 1, 100, 1e4, 1e6]

def rbf_kernel(c_list, q):
    error_list = [] # List of tuples (c, e_in, e_out)
    x_test1, y_test1 = get_1v5(x_test, y_test)
    x_train1, y_train1 = get_1v5(x_train, y_train)
    for c in c_list:
        clf = svm.SVC(kernel='rbf', C=c, degree=q, gamma=1.0)
        clf.fit(x_train1, y_train1)
        error_list.append((c, error(clf, x_train1, y_train1), error(clf, x_test1, y_test1))) 
    return error_list

results = rbf_kernel(cs, 2)
print("C: \t \t E_in:")
for i in range(len(results)):
    if results[i][0] > 100:
        print(f"{results[i][0]} \t {results[i][1]}")
    else:
        print(f"{results[i][0]} \t\t {results[i][1]}")


# ### Question 10: C - C = 100

# In[34]:


print("C: \t \t E_out:")
for i in range(len(results)):
    if results[i][0] > 100:
        print(f"{results[i][0]} \t {results[i][2]}")
    else:
        print(f"{results[i][0]} \t\t {results[i][2]}")

