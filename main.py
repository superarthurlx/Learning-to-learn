import tensorflow as tf 
import tensorlayer as tl 
import numpy as np 
import matplotlib.pyplot as plt 


def f(W, y, x):
    return np.sum(W@x - y)

