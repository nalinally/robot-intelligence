import numpy as np

def ReLU(x):
  return [np.max(x_, 0) for x_ in x]

def ReLU_diff(x):
  return [1 if x_ > 0 else 0 for x_ in x]

def equal(x):
  return x

def equal_diff(x):
  return [1 for _ in x]
  
def SSE(x, y):
  e = np.array(x) - np.array(y)
  return np.sum(e * e) / 2

def SSE_diff(x, y):
  e = np.array(x) - np.array(y)
  return e