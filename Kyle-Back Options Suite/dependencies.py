#!/usr/bin/env python
# coding: utf-8

# # Notice

# This .ipynb file contains dependencies for another notebook which is used to run simulations of the Kyle-Back Insider Trading model and also options pricing/IV curve fitting within said model.
# 
# Usage:
# 1. Open kyleback.ipynb and dependencies.ipynb
# 2. Download this file as a .py file
# 3. Place the .py file in the same folder as kyleback.ipynb
# 4. Run kyleback.ipynb normally

# # Packages

# In[26]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# # Black-Scholes European Option Formulas:

# In[3]:


np.seterr(all='warn')

def dee(T, K, r, sigma, S_0, pm):
    if pm == True:
        try:
            val = (np.log(S_0/K) + (r+(sigma**2/2))*T)/(sigma*np.sqrt(T))
        except OverflowError:
            if (np.log(S_0/K) + (r+(sigma**2/2))*T) > 0:
                val = np.inf
            else:
                val = -1*np.inf
    else:
        try:
            val = (np.log(S_0/K) + (r-(sigma**2/2))*T)/(sigma*np.sqrt(T))
        except OverflowError:
            if (np.log(S_0/K) + (r+(sigma**2/2))*T) > 0:
                val = np.inf
            else:
                val = -1*np.inf
    #print(val)
    return val

def BlackScholesMerton(T, K, r, sigma, S_0, option_type):
    try:
        d_up = dee(T,K,r,sigma,S_0,True)
        d_down = dee(T,K,r,sigma,S_0,False)
        val = 0

        if option_type == "call":
            val = S_0*norm.cdf(d_up) - K*np.exp(-r*T)*norm.cdf(d_down)
        elif option_type == "covered_call":
            val = S_0 - (S_0*norm.cdf(d_up) - K*np.exp(-r*T)*norm.cdf(d_down))
        else:
            val = K*np.exp(-r*T)*norm.cdf(-1*d_down) - S_0*norm.cdf(-1*d_up)
    except ZeroDivisionError:
        print("Raised ZeroDivisionError")
        if option_type == "call":
            val = np.maximum(S_0 - K,0)
        elif option_type == "covered_call":
            val = np.minimum(S_0,K)
        else:
            val = np.maximum(K - S_0,0)
            
    #print(val)
    return val


# # Option Greeks

# These are formulas for quickly computing Black-Scholes greeks. Vega is useful for computing IV using Newton's method.

# In[5]:


def delta(T, K, r, sigma, S_0, option_type):
    d_up = dee(T,K,r,sigma,S_0,True)
    if option_type == "call":
        val = norm.cdf(d_up)
    else:
        val = norm.cdf(d_up) - 1
    
    if np.isnan(val):
        val = 1
    return val

def gamma(T, K, r, sigma, S_0):
    d_up = dee(T,K,r,sigma,S_0,True)
    return np.exp(-r*T)*norm.pdf(d_up)/(S_0*sigma*np.sqrt(T))

def theta(T, K, r, sigma, S_0, option_type):
    d_down = dee(T,K,r,sigma,S_0,False)
    if option_type == "call":
        val = (S_0*norm.pdf(d_up)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d_down)
    else:
        val = (S_0*norm.pdf(d_up)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(d_down)
    return val
        
def vega(T, K, r, sigma, S_0):
    #Determine the vega of an option with the given specifications
    d_up = dee(T,K,r,sigma,S_0,True)
    return np.exp(-r*T)*S_0*np.sqrt(T)*norm.pdf(d_up)
    


# # Implied Volatility

# Here are various methods for computing the Black-Scholes implied volatilty of an option.

# ## Newton + Regula-Falsi

# This method needs to be worked on, it is not that well-written...

# In[ ]:


def impVol(option_price, T, K, r, S_0, option_type, starting_sigma, iterations):
    #Use the closed-form BSM formula and Newton's method to estimate the implied volatility of an option
    #with the given specifications. 
    
    #If Newton's method stops working, use Regula Falsi method instead.
    
    std = starting_sigma/2
    new_std = starting_sigma
    
    sup = 2*starting_sigma
    inf = 0.25*starting_sigma
    
    mode = "Newton"
    for i in range(iterations):
        old_std = std
        std = new_std
        
        if mode == "Newton":
            #if vega(T, K, r, std, S_0) <= 0:
                #print("Warning: Vega is not positive!")
            new_std = std - (BlackScholesMerton(T, K, r, std, S_0, option_type) - option_price)/vega(T, K, r, std, S_0)
            if new_std > std:
                inf = np.maximum(std, inf)
            else:
                sup = np.minimum(std, sup)
            if new_std < 0:
                mode = "Regula Falsi"
                #print("Switching to Regula Falsi")
        else:
            new_std = 0.5*(sup + inf)
            #print("Testing new std: " + str(new_std))
            if BlackScholesMerton(T, K, r, new_std, S_0, option_type) > option_price:
                sup = new_std
            else:
                inf = new_std
        #print("After iteration " + str(i) + "  (inf, sup) = (" + str(np.round(inf,5)) + "," + str(np.round(sup,5)) + ")")
        #print("After iteration " + str(i) + " new_std = " + str(new_std))
    
    if mode == "Newton":
        std = new_std
    else:
        std = 0.5*(sup + inf)
    
    return std


# ## Brent's Method

# Use Brent's method to compute implied volatility. Switch to a secant method step if a Brent's method step doesn't work.

# In[6]:


def threeSame(s1,s2,s3):
    a = np.array([s1 - s2, s1 - s3, s2 - s3])
    val = False
    if np.shape(np.where(a == 0)[0])[0] > 0:
        val = np.where(a == 0)[0][0]
        #print(val)
    return val

def impVolBrent(option_price, T, K, r, S_0, option_type, starting_sigma, iterations):
    #Use Brent's method to compute the implied volatility of an option
    std_0 = starting_sigma/2
    std_1 = 2*starting_sigma
    std_2 = starting_sigma
    
    for i in range(iterations):
        
        if threeSame(std_0, std_1, std_2) == False:
            g_0 = np.double(BlackScholesMerton(T, K, r, std_0, S_0, option_type) - option_price)
            g_1 = np.double(BlackScholesMerton(T, K, r, std_1, S_0, option_type) - option_price)
            g_2 = np.double(BlackScholesMerton(T, K, r, std_2, S_0, option_type) - option_price)

            sum_0 = (std_2*g_1*g_0)/((g_2 - g_1)*(g_2 - g_0))
            sum_1 = (std_1*g_0*g_2)/((g_1 - g_0)*(g_1 - g_2))
            sum_2 = (std_0*g_1*g_2)/((g_0 - g_1)*(g_0 - g_2))

            new_std = sum_0 + sum_1 + sum_2
            
        elif threeSame(std_0, std_1, std_2) == 0 or threeSame(std_0, std_1, std_2) == 2:
            print("Warning! Two or more points are the same!")
            g_0 = BlackScholesMerton(T, K, r, std_0, S_0, option_type) - option_price
            g_1 = BlackScholesMerton(T, K, r, std_2, S_0, option_type) - option_price
            new_std = std_2 - g_0*(std_2 - std_0)/(g_1 - g_0)
        elif threeSame(std_0, std_1, std_2) == 1:
            print("Warning! Two or more points are the same!")
            g_0 = BlackScholesMerton(T, K, r, std_1, S_0, option_type) - option_price
            g_1 = BlackScholesMerton(T, K, r, std_2, S_0, option_type) - option_price
            new_std = std_2 - g_0*(std_2 - std_1)/(g_1 - g_0)
        
        #print(new_std)
        if math.isinf(new_std) == False and math.isnan(new_std) == False:
            std_0 = std_1
            std_1 = std_2
            std_2 = new_std
        else:
            #print("Number of iterations taken: " + str(i))
            #print("final std: " + str(std_2))
            return std_2
    #print("Number of iterations taken: " + str(i))
    #print("final std: " + str(new_std))
    return new_std


# ## Bisection method

# In[8]:


def impVolBisection(option_price, T, K, r, S_0, option_type, starting_sigma, iterations):
    init_val = BlackScholesMerton(T, K, r, starting_sigma, S_0, option_type) - option_price
    
    #The initial value will either be our "a" or our "b" depending whether it is above or below the option price
    #In rare cases the guess might be right on the money, so we need to account for that too.
    if init_val < 0:
        a = starting_sigma
        b = 2*starting_sigma
        while BlackScholesMerton(T, K, r, b, S_0, option_type) - option_price < 0:
            a = b
            b = 2*b
            #print("Updated guess interval to (" + str(a) + ", " + str(b) + ")")
        
    elif init_val > 0:
        a = 0.5*starting_sigma
        b = starting_sigma
        while BlackScholesMerton(T, K, r, a, S_0, option_type) - option_price > 0:
            b = a
            a = 0.5*a
            if b == 0:
                #This case might in occur in the event of significant numerical instability
                return 0
            #print("Updated guess interval to (" + str(a) + ", " + str(b) + ")")
    else:
        return starting_sigma
    
    #print("Setting initial guess interval to (" + str(a) + ", " + str(b) + ")")
    #Once our initial interval is set, it's time to do the iterations.
    for i in range(iterations):
        t = 0.5*(a+b)
        g = BlackScholesMerton(T, K, r, t, S_0, option_type) - option_price
        
        if g < 0:
            a = t
        elif g > 0:
            b = t
        else:
            return t
        
        #print("Guess after iteration " + str(i) + ": " + str(0.5*(a+b))) 
        
    return 0.5*(a+b)


# ## Newton-Bisection

# Try Newton's method, then switch to bisection method if Newton's method produces a "bad" step.

# In[ ]:


def impVolNewtonBisection(option_price, T, K, r, S_0, option_type, starting_sigma, iterations):
    #Attempt to use Newton's method iterations to find implied volatility.
    #Switch to bisection method during moments of instability.
    
    #First, let's try and find a good guess for an interval that the implied volatility could lie in
    init_val = BlackScholesMerton(T, K, r, starting_sigma, S_0, option_type) - option_price
    
    #The initial value will either be our "a" or our "b" depending whether it is above or below the option price
    #In rare cases the guess might be right on the money, so we need to account for that too.
    if init_val < 0:
        a = starting_sigma
        b = 2*starting_sigma
        while BlackScholesMerton(T, K, r, b, S_0, option_type) - option_price < 0:
            a = b
            b = 2*b
        
    elif init_val > 0:
        a = 0.5*starting_sigma
        b = starting_sigma
        while BlackScholesMerton(T, K, r, a, S_0, option_type) - option_price > 0:
            b = a
            a = 0.5*a
    else:
        return starting_sigma

    #print("Setting initial guess interval to (" + str(a) + ", " + str(b) + ")")
    std = (a + b)/2
    
    mode = "Newton"
    for i in range(iterations):
        #Attempt to run a Newton's method step
        #If the new guess does not lie in our interval, do a bisection method step instead
        new_std = std - (BlackScholesMerton(T, K, r, std, S_0, option_type) - option_price)/vega(T, K, r, std, S_0)
        if new_std <= b and a <= new_std:
            #Accept the Newton's method step
            std = new_std
        else:
            #Run a bisection method step instead
            #print("Error with Newton! Trying Bisection instead.")
            t = 0.5*(a+b)
            g = BlackScholesMerton(T, K, r, t, S_0, option_type) - option_price

            if g < 0:
                a = t
            elif g > 0:
                b = t
            else:
                #In the rare case our guess is on the money, return t immediately
                return t

            std = 0.5*(a+b)
    
    return std
    


# # Gaussian Mixture

# Surprisingly, I could not find anything within numpy or scipy which could natively handle the Gaussian mixture distribution. Nonetheless, it is not *too* difficult to define CDF and PDF functions for this case.

# In[9]:


def GaussianMixtureCDF(x, weights, means, volatilities):
    top = x.reshape(-1,1) - means
    argument = top/volatilities
    return np.sum(weights*norm.cdf(argument),axis=1)


# In[15]:


def GaussianMixturePDF(x, weights, means, volatilities):
    top = x.reshape(-1,1) - means
    #print(top)
    argument = top/volatilities
    #print(argument)
    return np.sum((weights/volatilities)*norm.pdf(argument),axis=1)


# This is for computing the PDF of a random variable $$X = exp(Y)$$ such that $$Y$$ is a Gaussian Mixture random variable with the given parameters.

# In[33]:


def LogGaussianMixturePDF(x,weights,means,volatilities):
    y = np.log(x)
    top = y.reshape(-1,1) - means
    #print(top)
    argument = top/volatilities
    #print(argument)
    return np.sum(((weights/volatilities)/y.reshape(-1,1))*norm.pdf(argument),axis=1)

