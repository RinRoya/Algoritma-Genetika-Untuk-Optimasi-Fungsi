#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


class GeneticAlgorithm():
    def __init__(self,n_chrom, chrom_length, min_value=0,
                 max_value=1, p_cross=0, p_mut=1, max_generate=100):
        self.n_chrom = n_chrom
        self.chrom_length = chrom_length
        self.min_value = min_value
        self.max_value = max_value
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.max_generate = max_generate
        
    def solve(self):
        result = generate(self.n_chrom, self.chrom_length,
                        self.min_value, self.max_value,
                        self.p_cross, self.p_mut, self.max_generate)
        return result


# In[3]:


def createPopulation(n_chrom, chrom_length, min_value, max_value):
    pop = np.random.uniform(min_value, max_value, size=(n_chrom, chrom_length))
    pop = pd.DataFrame(pop)
    pop.columns = ["x"+str(i) for i in range(1, chrom_length+1)]
    return pop


# In[4]:


def fitness(pop):
    fitness = (pop.x1 + 2*pop.x2 - 7)**2 + (2*pop.x1 + pop.x2 - 5)**2 + (pop.x2 + pop.x3)**2
    pop["fitness"] = fitness
    return pop


# In[5]:


def selection(n_chrom):
    position = np.random.permutation(n_chrom)
    return position[0], position[1]


# In[6]:


def crossover(pop, n_chrom):
    temp_pop = pop.copy()
    for i in range(n_chrom):
        s1, s2 = selection(n_chrom)
        cross = (pop.loc[s1] + pop.loc[s2])/2
        temp_pop.loc[i] = cross
    return temp_pop


# In[7]:


def mutation(pop):
    pop += np.random.uniform(-0.2,0.2)
    return pop


# In[8]:


def combine(pop, popc, popm):
    temp_pop = pop.append([popc, popm], ignore_index=True)
    temp_pop.drop_duplicates(inplace=True)
    return temp_pop


# In[9]:


def sort_pop(pop):
    pop.sort_values(by="fitness", inplace=True)
    pop.index = range(len(pop))
    return pop


# In[10]:


def elimination(pop, n_chrom):
    return pop.head(n_chrom)


# In[11]:


def generate(n_chrom, chrom_length, min_value, max_value,p_cross, p_mut, max_generate):
    pop = createPopulation(n_chrom, chrom_length, min_value, max_value)
    pop = fitness(pop)
    for i in range(max_generate):
        pc = np.random.rand()
        pm = np.random.rand()

        popc = pd.DataFrame()
        popm = pd.DataFrame()

        if pc < p_cross:
            popc = crossover(pop, n_chrom)
            popc = fitness(popc)

        if pm < p_mut:
            popm = mutation(pop)
            popm = fitness(popm)

        pop = combine(pop, popc, popm)
        pop = sort_pop(pop)
        pop = elimination(pop, n_chrom)
    return pop.loc[0]

