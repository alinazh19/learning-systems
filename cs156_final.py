#!/usr/bin/env python
# coding: utf-8

# # Learning Systems - Final
# ## Name: Alina Zhang
# ## Email: h2hang@caltech.edu

# In[1]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC


# ### Question 1: E - None of the above
# Because you have to add up the dimensions of each layer in the binomial expansion model, the formula for finding the dimensionality of Q = n would be the sum of 2 -> n + 1, ie 65. Therefore, the dimensionality of the Z-space of the polynomial transform of Q = 10 of dimension d = 2 would be 65 (None of the above). 

# ### Question 2: D - H is the logistic regression model
# [A] - If H has one hypothesis, then that hypothesis would always be chosen, no matter the dataset. Therefore, the training will still only result in one hypothesis, and a $\bar g \in \mathcal H$.   
# [B] - If H is the set of all real-valued, constant hypotheses, then the chosen hypothesis would also be a real-valued constant b. Therefore, $\bar g \in \mathbb R$, which means that $\bar g \in \mathcal H$.   
# [C] - If H is the linear regression model, then all of the hypotheses are polynomials with real-valued coefficients, which means that the values of $g^{(D)}$ will also all have real coefficients. When we take the expected value of $g^{(D)}$ to with respect to $D$ to find $\bar g$, $\bar g \in \mathcal H$ since that set is also all real valued.   
# [D] - If H is the logistic regression model, the hypotheses are sigmoidal functions. When we find the expected value of $g^{(D)}$ with respect to D, we know that $\bar g$ is not necessarily a sigmoid. This is because the sum of two sigmoidal functions don't necessarily have to be sigmoids. Therefore, $\bar g \not\in \mathcal H$.   
# [E] - Since D is the correct answer, this one is incorrect. 

# ### Question 3: D - We can always determine if there is overfitting by comparing the values of $(E_{out} - E_{in})$
# [A] - This is true, we overfit when we have multiple hypotheses and are tempted into choosing one because the value of $E_in$ is higher than another.    
# [B] - This is true, we overfit when we have multiple hypotheses and choose one because the value of $E_{in}$ is the lowest, while the value of $E_{out}$ is higher in comparison.   
# [C] - This is true, and is basically the definition for overfitting. We are choosing a hypothesis that has a deceptively low value of $E_{in}$, while their $E_{out}$ is much higher. This directly correlates with multiple hypotheses with different values of $E_{out} - E_{in}$.   
# [D] - This is FALSE, because we need multiple hypotheses to compare the value of $E_{in} - E_{out}$ with. There are multiple scnearios in which two hypotheses can have the same values of $E_{out}$ and $E_{in}$ and it is only considered overfitting depending on other factors.  
# [E] - This is true, because you need to compare multiple hypotheses.      

# ### Question 4: D - Stochastic noise does not depend on the hypothesis set. 
# [A] - This is false, there is no reason why deterministic noise cannot occur with stochastic noise.   
# [B] - This is false, deterministic noise depends on the hypothesis set, as some models approximate $f$ better than others.    
# [C] - This is false, because if we assume H is fixed, the complexity of f changes the deterministic noise. If we increase the complexity of f, the deterministic noise also increases.   
# [D] - This is true, because stochastic noise is just randomness in the data and therefore does not depend on the hypotheses. 
# [E] - This is false, since the stochastic noise is dependent on how the target function randomly generates the dataset. If the target function changes, then the stochastic noise also changes.   

# ### Question 5: A - $w_{reg} = w_{lin}$
# Since we already know that our linear regression solution is constrained, we know that $w_{reg} = w_{lin}$. $w_{lin}$ is also found by minimizing the same constraint, so it makes sense that they would be essentially the same in this scenario. 

# ### Question 6: B - translated into augmented error
# [A] - This is false, because hard-order constraints utilize a lower-order model, while soft-order constraints constrain the parameters of the model, both of which can't be translated into one another. 
# [B] - This is true, we are able to translate soft-order constraints into augmented error. By using constraints, we can regularize the hypothesis into augmented error through a series of calculations. This can be done using weight decay regularization.   
# [C] - This is false, though regularization affects the VC model such that we need to look at an "effective VC dimension" instead.    
# [D] - This is false, because while regularization make $E_{in}$ higher, it will approximate the target function better and as a result make $E_{out}$ lower.    
# [E] - This is false, because soft-order constraints can be translated into augmented error, so B is correct. 

# ### Question 7: D - 8 versus all

# In[2]:


def get_data(filename):
    x = []
    y = []
    f = open(filename, "r")
    for line in f:
        s = line.split()
        x.append([float(s[1]), float(s[2])])
        y.append(float(s[0]))
    return x, y    

def numvall(Y, num):
    data = []
    for i in Y:
        if i == num:
            data.append(1)
        else:
            data.append(-1)
    return data
    
def numvnum(X, Y, num1, num2):
    x_lst = []
    y_lst = []
    for i in range(len(Y)):
        if Y[i] == num1:
            x_lst.append(X[i])
            y_lst.append(1)
        elif Y[i] == num2:
            x_lst.append(X[i])
            y_lst.append(-1)
    return x_lst, y_lst

def ft(data, to_transform):
    transformed_data = []
    for x1, x2 in data:
        if to_transform:
            transformed_data.append((1, x1, x2, x1*x2, x1**2, x2**2))
        else:
            transformed_data.append((1, x1, x2))
    return transformed_data

def w_reg(Z, y, ld):
    ztz = np.dot(np.transpose(Z), Z) + ld * np.identity((np.dot(np.transpose(Z), Z)).shape[0])
    x = np.dot(np.linalg.inv(ztz), np.transpose(Z))
    return np.dot(x, y)

def error(weight, data, Y):
    sum_error = 0
    for x, y in zip(data, Y):
        hypothesis = np.sign(np.dot(weight, x))
        if (hypothesis != y):
            sum_error += 1
    return sum_error / len(Y)


# In[3]:


LAMBDA = 1
def question_7(num, ld, transform):
    # Get data from file
    x_test, y_test = get_data("features.test")
    x_train, y_train = get_data("features.train")

    # Edits the data to be num vs all
    y_test = numvall(y_test, num)
    y_train = numvall(y_train, num)

    # Transform
    phi_train, phi_test = ft(x_train, transform), ft(x_test, transform)
    weight = w_reg(phi_train, y_train, ld)
    
    # Get error
    e_in = error(weight, phi_train, y_train)
    e_out = error(weight, phi_test, y_test)
    return e_in, e_out

for n in range(5, 10):
    print("num:", n, "\t\tE_in:", round(question_7(n, LAMBDA, False)[0], 2))


# ### Question 8: B - 1 versus all

# In[4]:


for n in range(0, 5):
    print("num:", n, "\t\tE_out:", round(question_7(n, LAMBDA, False)[1], 2))


# ### Question 9: E - The transform improves the out-of-sample performance of '5 versus all', but by less than 5%.

# In[5]:


print("E_out values:")
for n in range(0, 10):
    trans = question_7(n, LAMBDA, True)
    no_trans = question_7(n, LAMBDA, False)  
    print(f"num: {n} \ttrans: {round(trans[1], 6)} \tnot trans: {round(no_trans[1], 6)}")


# ### Question 10: A - Overfitting occurs (from λ = 1 to λ = 0.01).
# Because E_in decreases while E_out increases, going from λ = 1 to λ = 0.01 is overfitting. 

# In[6]:


def question_10(num1, num2, ld, transform):
    # Get data from file
    x_test, y_test = get_data("features.test")
    x_train, y_train = get_data("features.train")
    
    # Edits the data to be num vs num
    x_test, y_test = numvnum(x_test, y_test, num1, num2)
    x_train, y_train = numvnum(x_train, y_train, num1, num2)

    # Transform
    phi_train, phi_test = ft(x_train, transform), ft(x_test, transform)
    weight = w_reg(phi_train, y_train, ld)
    
    # Get error
    e_in = error(weight, phi_train, y_train)
    e_out = error(weight, phi_test, y_test)
    return e_in, e_out

for ld in [1, 0.01]:
    sol = question_10(1, 5, ld, True)
    print(f"lambda: {ld} \tE_in: {round(sol[0], 3)} \tE_out: {round(sol[1], 3)}")


# ### Question 11: C - 1, 0, -0.5
# By looking at the graph, we can see that the classification line is at z1 = 0.5 as z2 makes no impact on whether the points are classified as red or green. $$ (a, b)\cdot (0.5, y) = 0.5 $$ We have the right hand side as 0.5 since the answer choices all have -0.5 for b. The only answer choice that could work is where (a, b) is (1, 0). Since a must be 1, we are left with B and C, but B can't work since y can be anything. Therefore, the correct answer choice is C. 

# In[7]:


x = [(1,0), (0,1), (0,-1), (-1,0), (0,2), (0,-2), (-2,0)]
y = [-1, -1, -1, 1, 1, 1, 1]

# Finds the z-space
def z_space(x):
    z_list = []
    for x1, x2 in x:
        z_list.append((x2 ** 2 -  2 * x1 - 1, x1 ** 2 - 2 * x2 + 1))
    return z_list

z = z_space(x)
print(z)

# Separates the dots according to y
for i in range(0,4):
    plt.scatter(z[i][0], z[i][1], c='red')
for i in range(3, 7):
    plt.scatter(z[i][0], z[i][1], c='green')
plt.show()


# ### Question 12: C - 4-5

# In[8]:


machine = svm.SVC(C=1000, kernel="poly", degree=2, coef0=1)
machine.fit(X=x, y=y)

print(f"the number of support vectors is {sum(list(machine.n_support_))}.")


# ### Question 13: A - $\leq 5\%$ of the time

# In[9]:


# Helper functions
POINTS = 100

def tf(x1, x2):
    return np.sign(x2 - x1 + 0.25*math.sin(math.pi*x1))

def gen_points(n):
    x = []
    y = []
    for i in range(n):
        pt = [random.uniform(-1, 1), random.uniform(-1, 1)]
        x.append(pt)
        y.append(tf(pt[0], pt[1]))
    return (np.array(x), np.array(y))
    
def svm_rbf(x,y, c, gamma): # Returns E_in, clf, and # of support vectors
    clf = svm.SVC(kernel='rbf', C=c, gamma=gamma)
    clf.fit(x, y)
    y_hat = clf.predict(x)
    return np.sum(y * y_hat < 0) / (y.size), clf, clf.support_vectors_.shape[0]


# In[10]:


RUNS = 1000
GAMMA = 1.5
C = 10e4

def question_13(runs):
    E_in = []
    for i in range(runs): 
        (x,y) = gen_points(POINTS)
        r = svm_rbf(x, y, C, GAMMA)[0]
        if (r != 0):
            E_in.append(r)
    return(len(E_in) / runs)
print(f"You get a dataset that is not separable by the RBF kernel {question_13(RUNS)*100}% of the time")


# ### Question 14: E - $>75\%$ of the time

# In[11]:


# Helper functions
K = 9
RUNS = 100
GAMMA = 1.5
C = 1e4

def rbf_func(x, mu0, gamma):
    return np.exp(-gamma * np.sum((x - mu0) ** 2))

def rbf(x, mu, gamma):
    mu_size = mu.shape[0]
    x_size = x.shape[0]
    z = np.empty((x_size, 1 + mu_size))
    z[:, 0] = np.ones(x_size)
    for i in range(mu_size):
        z[:, i + 1] = np.apply_along_axis(rbf_func, 1, x, mu[i], gamma)
    return z

def set_pos(x1, mu):
    return np.argmin(np.sqrt(np.sum((x1 - mu) ** 2, axis=1)))

def lloyd(x, k):
    # Randomly choose k centers
    centers = []
    for i in range(k):
        pt = [random.uniform(-1, 1), random.uniform(-1, 1)]
        centers.append(pt)
    centers = np.array(centers)
    
    # Iterate
    size = x.shape[0]
    prev_cluster = np.ones(size)
    curr_cluster = np.zeros(size)
    while not np.all(prev_cluster == curr_cluster):
        prev_cluster = curr_cluster
        curr_cluster = np.apply_along_axis(set_pos, 1, x, centers) # Assign item to closest cluster    
        for g in np.unique(curr_cluster).tolist():    
            centers[g] = np.mean(x[curr_cluster == g], axis=0) # Find new mean for each cluster
            
    # If clusters become empty, discard the run and repeat
    if len(np.unique(curr_cluster)) != k: 
        (x, y) = gen_points(size) 
        lloyd(x, k)
    return centers

def reg_rbf(x, y, gamma, k):
    c = lloyd(x, k)
    z = np.empty((x.shape[0], 1 + c.shape[0]))
    z[:,0] = np.ones(x.shape[0])
    for i in range(c.shape[0]):
        z[:,i+1] = np.apply_along_axis(rbf_func, 1, x, c[i],gamma)
    w = np.dot(np.dot(np.matrix(np.dot(np.transpose(z), z) + 0 * np.eye(z.shape[1])).getI(), np.transpose(z)), y).getA()[0,:]
    yhat = np.dot(z,w)
    e_in = np.sum(yhat*y<0)/(1.*y.size)
    return c, w, e_in


# In[12]:


def question_14(runs, k): # Returns e_out for regular, kernel, and how often kernel beats out regular
    Eout_reg = []
    Eout_ker = []
    count = 0
    for i in range(runs):
        (x, y) = gen_points(POINTS)
        (x_out, y_out) = gen_points(POINTS)
        
        # Find E_out for regular RBF
        c, w, e_in = reg_rbf(x, y, GAMMA, k)
        z = rbf(x_out, c, GAMMA)
        y_hat = np.dot(z, w) 
        reg_out = np.sum(y_hat * y_out < 0) / y_out.size
        Eout_reg.append(reg_out)
        
        # Find E_out for kernel RBF
        _, clf, sv = svm_rbf(x,y, C, GAMMA)
        y_hat = clf.predict(x_out)
        ker_out = np.sum(y_hat * y_out < 0) / y_out.size
        Eout_ker.append(ker_out)
        if ker_out < reg_out:
            count += 1
    return Eout_reg, Eout_ker, count / runs
print(f"The kernel form beats the regular form {question_14(RUNS, K)[2] * 100}% of the time.")


# ### Question 15: D - $>60\%$ but $\leq 90\%$ of the time

# In[13]:


K = 12
GAMMA = 1.5
print(f"For K = 12, the kernel beats regular {question_14(RUNS, K)[2] * 100}% of the time.")


# ### Question 16: D - both $E_{in}$ and $E_{out}$ go down

# In[14]:


RUNS = 100
GAMMA = 1.5
POINTS = 100

def question_16(runs, k):
    lst_ein = []
    lst_eout = []
    for i in range(runs):
        x_in, y_in = gen_points(POINTS)
        x_out, y_out = gen_points(POINTS)
        mu, w, e_in = reg_rbf(x_in, y_in, GAMMA, k)
        y_hat = np.dot(rbf(x_out, mu, GAMMA), w)
        lst_ein.append(e_in)
        lst_eout.append(np.sum(y_hat * y_out < 0) / y_out.size)
    return lst_ein, lst_eout

ein_9, eout_9 = question_16(RUNS, 9)
ein_12, eout_12 = question_16(RUNS, 12)

# E_in:
count_in = 0
for i in range(len(ein_9)):
    if ein_9[i] > ein_12[i]:
        count_in += 1
        
# E_out:
count_out = 0
for i in range(len(ein_9)):
    if eout_9[i] > eout_12[i]:
        count_out += 1       
        
print(f"The number of runs where E_in goes down from K = 9 to K = 12 is {count_in} and up is {len(ein_9) - count_in}.")
print(f"The number of runs where E_out goes down from K = 9 to K = 12 is {count_out} and up is {len(eout_9) - count_out}.")


# ### Question 17: C - both $E_{in}$ and $E_{out}$ go up

# In[15]:


RUNS = 100
K = 9
POINTS = 100

def question_17(runs, gamma):
    lst_ein = []
    lst_eout = []
    for i in range(runs):
        x_in, y_in = gen_points(POINTS)
        x_out, y_out = gen_points(POINTS)
        mu, w, e_in = reg_rbf(x_in, y_in, gamma, K)
        y_hat = np.dot(rbf(x_out, mu, gamma), w)
        lst_ein.append(e_in)
        lst_eout.append(np.sum(y_hat * y_out < 0) / y_out.size)
    return lst_ein, lst_eout

ein_1, eout_1 = question_17(RUNS, 1.5)
ein_2, eout_2 = question_17(RUNS, 2)

# E_in:
count_in = 0
for i in range(len(ein_1)):
    if ein_1[i] > ein_2[i]:
        count_in += 1
        
# E_out:
count_out = 0
for i in range(len(ein_2)):
    if eout_1[i] > eout_2[i]:
        count_out += 1       
        
print(f"The number of runs where E_in goes down from K = 9 to K = 12 is {count_in} and up is {len(ein_1) - count_in}.")
print(f"The number of runs where E_out goes down from K = 9 to K = 12 is {count_out} and up is {len(eout_1) - count_out}.")


# ### Question 18: A - $\leq 10\%$ of the time

# In[17]:


RUNS = 100
K = 9
GAMMA = 1.5
POINTS = 100

def question_18(runs):
    x_in, y_in = gen_points(POINTS)
    x_out, y_out = gen_points(POINTS)
    count = 0
    for _ in range(runs): # c is massive for hard-margin svm
        _, _, e_in = reg_rbf(x_in, y_in, GAMMA, K)
        if (e_in == 0):
            count += 1
    return count / runs
print(f"The regular RBF achieves E_in = 0 {question_18(RUNS) * 100}% of the time.")


# ### Question 19: B - The posterior increases linearly over [0, 1]. 
# The formula that we can use is $$P(h = f | D) = \frac{P(D|h = f) P (h = f)}{P(D)}$$ and find what $P(h = f | D)$ is, since that is the posterior probability that $h = f$ given the sample point S. Since we know that $P(h = f)$ is uniform over [0, 1], and that $P(D|h = f)$ is a linear distribution increasing over [0, 1], we know that the left hand side, $P(h = f | S)$ also increases linearly because the other terms are either uniform or constant. We know that $P(D|h = f)$ is linear because for every person that has a heart attack, the probability that any person who got a heart attack increases with h, which means that as h increases $P(D|h = f)$ also increases. 

# ### Question 20: C - $E_{out}(g)$ cannot be worse than the average of $E_{out}(g1)$ and $E_{out}(g2)$.
# [A] - False: if $g_2$ is consistently much different from the target function than $g_1$ is, $E_{out}(g)$ would be worse than $E_{out}(g1)$.    
# [B] - False: since $g$ is the average of $g_1$ and $g_2$, it would make sense that $E_{out}(g)$ is also in between the min and max between $E_{out}(g1)$ and $E_{out}(g2)$, and that it can definitely be worse than the smallest between them.    
# [C] - True: because we are using mean squared error, which means that $g$ is in between $g_1$ and $g_2$, the error of $E_{out}(g)$ cannot be worse than the average of $E_{out}(g1)$ and $E_{out}(g2)$.   
# [D] - False: if one of the functions was consistently overestimating and the other was consistently underestimating, g would do a better job of approximating the target function, and therefore $E_{out}(g)$ would be less than both $E_{out}(g1)$ and $E_{out}(g2)$.   
# [E] - False: C is true. 
# 
