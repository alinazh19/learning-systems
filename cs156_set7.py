#!/usr/bin/env python
# coding: utf-8

# # Learning Systems - Homework 7
# ## Name: Alina Zhang
# ## Email: h2hang@caltech.edu

# In[91]:


# Imports
import numpy as np
import math 
import random
import functools
from sklearn import svm
from sklearn.linear_model import Perceptron


# ### Question 1: D - k = 6

# In[2]:


# Helper functions
def transform(x1, x2, y, k):
    return ((1, x1, x2, x1**2, x2**2, x1*x2, abs(x1 - x2), abs(x1 + x2))[:k + 1], y)

def get_phi(data, k):
    transformed_data = []
    for x1, x2, y in data:
        transformed_data.append(transform(float(x1), float(x2), y, k))
    return transformed_data

def get_data(file):
    data = []
    file = open(file)
    for line in file:
        data.append([float(i) for i in line.split()])
    return data

def split(data):
    return (data[:25], data[25:]) # 25 first half, 10 second half

def get_lin_weight(data):
    Z, y = zip(*data)
    return np.dot(np.linalg.pinv(Z), y)

def get_error(weight, data):
    error = 0
    Z, Y = zip(*data)
    for i in range(len(Y)):
        hypothesis = np.sign(np.dot(weight, Z[i]))
        if (hypothesis != Y[i]):
            error += 1
    return error / len(Y)


# In[3]:


def lin_reg(k, f):
    # Get data from file & transform
    data_in, data_out = get_data("cs156_in.txt"), get_data("cs156_out.txt")
    phi_in, phi_out = get_phi(data_in, k), get_phi(data_out, k)

    # Find training set and get lin weight
    if f == True:
        training, validation = split(phi_in)
    else:
        validation, training = split(phi_in)
    lin_weight = get_lin_weight(training)
    
    # Find errors
    val_error = get_error(lin_weight, validation)
    out_error = get_error(lin_weight, phi_out)
    return val_error, out_error

for k in range(3, 8):
    print("k:", k, "| validation error:", lin_reg(k, True)[0])


# ### Question 2: E - k = 7

# In[4]:


for k in range(3, 8):
    print("k:", k, "| out of sample error:", lin_reg(k, True)[1])


# ### Question 3: D - k = 6

# In[5]:


for k in range(3, 8):
    print("k:", k, "| validation error:", lin_reg(k, False)[0])


# ### Question 4: D - k = 6

# In[6]:


for k in range(3, 8):
    print("k:", k, "| out of sample error:", lin_reg(k, False)[1])


# ### Question 5: B - 0.1, 0.2

# In[7]:


print("The out of sample error for question 1 is", lin_reg(6, True)[1], "and question 2 is", lin_reg(6, False)[1])


# ### Question 6: D - 0.5, 0.5, 0.4

# In[8]:


def v_bias_exp():
    e1, e2 = random.uniform(0, 1), random.uniform(0, 1)
    e = min([e1, e2])
    return e1, e2, e

e1_lst = []
e2_lst = []
e_lst = []
for i in range(100000):
    e1, e2, e = v_bias_exp()
    e1_lst.append(e1)
    e2_lst.append(e2)
    e_lst.append(e)
    
print(np.average(e1_lst), np.average(e2_lst), np.average(e_lst))


# ### Question 7: C

# In[28]:


def rho(q):
    if q == 'a':
        return np.sqrt(np.sqrt(3) + 4)
    elif q == 'b':
        return np.sqrt(np.sqrt(3) - 1)
    elif q == 'c':
        return np.sqrt(9 + 4 * np.sqrt(6))
    else:
        return np.sqrt(9 - np.sqrt(6))

def get_val(q):
    return np.array([-1, rho(q), 1]), np.array([0, 1, 0]), np.array([0, 1, 0]).shape[0]

def const_crossval(q):
    X, Y, N = get_val(q)
    error_sum = 0
    for i in range(N):
        x_t, y_t = np.delete(X, i), np.delete(Y, i)
        n_t = y_t.shape[0]
        x, y, n = X[i], Y[i], 1
        
        # Constant model
        b = sum(y_t) / n_t
        p = np.repeat(b, n_t)
        
        # Squared error measure
        error_sum += np.power(y - b, 2)
        return np.average(error_sum)

def linear_crossval(q):
    X, Y, N = get_val(q)
    error_sum = 0
    for i in range(N):
        x_t, y_t = np.delete(X, i)[:, None], np.delete(Y, i)[:, None]
        n_t = y_t.shape[0]
        x, y, n = X[i], Y[i], 1
        
        # Linear model
        x_c = np.c_[np.ones(n_t), x_t]
        x_v = [1, x]
        x_dagger = np.dot(np.linalg.inv(np.dot(x_c.T, x_c)), x_c.T)
        w = np.dot(x_dagger, y_t)

        # Squared error measure
        pred_val = np.dot(x_c, w)
        error_sum += np.power(y - pred_val, 2) 
        return np.average(error_sum)
print("the squared error measure of the constant model for choice c is", const_crossval('c'), "and the linear model is", linear_crossval('c'))


# ### Question 8: C - 60%

# In[101]:


# Helper functions
RUNS = 1000
N = 10

def generate_data(N):
    p1 = np.random.uniform(-1, 1, 2)
    p2 = np.random.uniform(-1, 1, 2)
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = p2[1] - slope * p1[0]
    
    def classify(X):
        x, y = X
        if y > slope * x + c: 
            return 1
        else:
            return -1
    
    X = np.c_[np.random.uniform(-1, 1, N), np.random.uniform(-1, 1, N)]
    Y = np.apply_along_axis(classify, 1, X)
    return (X, Y, y)

def generate_test_data(N, y):
    X = np.c_[np.random.uniform(-1, 1, N), np.random.uniform(-1, 1, N)]
    Y = np.apply_along_axis(y, 1, X)
    return (X, Y)


# In[94]:


def perceptron(N): 
    X, Y, y = generate_data(N)
    clf = Perceptron()
    clf.fit(X, Y)
    x, y = generate_test_data(100 * N, y)
    return clf.score(x, y)

def SVM(N): 
    X, Y, y = generate_data(N)
    clf = svm.SVC(kernel='linear', C=np.Inf, cache_size=20000)
    clf.fit(X, Y)
    x, y = generate_test_data(100 * N, y)
    return clf.score(x, y)

score = 0
for i in range(RUNS):
    if SVM(N) > perceptron(N):
        score += 1
results = score / RUNS

print("svm is better than pla in approximating f", round(results, 1), "of the time.")


# ### Question 9: D - 65%

# In[108]:


sv = []
RUNS = 1000
count = 0

for i in range(RUNS):
    X, Y, y = generate_data(N)
    clf_ = Perceptron()
    clf_.fit(X, Y)
    x, y = generate_test_data(100 * N, y)
    pla_score = clf_.score(x, y)
    clf = svm.SVC(kernel='linear', C=np.Inf, cache_size=20000)
    clf.fit(X, Y)
    x, y = generate_test_data(100 * N, y)
    svc_score = clf.score(x, y)
    sv.append(len(clf.support_vectors_))
    if SVM(N) > perceptron(N):
        score += 1

results = score / RUNS

print("svm is better than pla in approximating f", round(results, 1), "of the time" )


# ### Question 10: B - 3

# In[109]:


print("the average number of support vectors is", round(sum(sv)/float(len(sv)), 1))

