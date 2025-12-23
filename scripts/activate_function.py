
def ReLU_y(x):
  return max(x, 0)

def ReLU_dx(x):
  if x > 0:
    return 1
  else:
    return 0

def one(x):
  return 1

def one_diff(x):
  return 0