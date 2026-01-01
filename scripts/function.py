import numpy as np
rng = np.random.default_rng()

#  activate
def ReLU(x):
  return [np.max(x_, 0) for x_ in x]

def ReLU_diff(x):
  return np.diag([1 if x_ > 0 else 0 for x_ in x])

def LeakyReLU(a):
  def func(x):
    return [np.max([x_, a * x_]) for x_ in x]
  return func

def LeakyReLU_diff(a):
  def func(x):
    # print(f"[leakyrelu] {x} {np.diag([1 if x_ > 0 else a for x_ in x])}")
    return np.diag([1 if x_ > 0 else a for x_ in x])
  return func

def softMax(a):
  def func(x):
    exp_vec = np.exp(a * x)
    return exp_vec / np.sum(exp_vec)
  return func

def softMax_diff(a):
  f = softMax(a)
  def func(x):
    y = f(x)
    return a * (np.diag(y) - y[:, None] @ y[None, :])
  return func

def equal(x):
  return x

def equal_diff(x):
  return np.eye(len(x))
  
# normalize
def fixed_normalize(x):
  epsilon = 1e-5
  return (x - np.mean(x)) / np.sqrt(np.var(x) + epsilon)

def fixed_normalize_diff(x):
  y = fixed_normalize(x)
  n = float(len(x))
  sigma = np.std(x)
  # print(f"{np.eye(len(x))} {y[:, None] @ y[None, :]} {(1 + (y[:, None] @ y[None, :])) / n} {sigma} {(np.eye(len(x)) - ((1 + (y[:, None] @ y[None, :])) / n)) / sigma}")
  return (np.eye(len(x)) - ((1 + (y[:, None] @ y[None, :])) / n)) / sigma

# pooling
def average_pooling(dx, dy):
  def func(data):
    return np.mean(np.mean([[data[i::dy, j::dx] for j in range(dx)] for i in range(dy)], axis=0), axis=0)
  return func

def index_pooling(dx, dy):
  def func(data):
    return data[::dx, ::dy]
  return func

# noise
def white_noise(rate, scope):
  def rand():
    return scope[0] + ((scope[1] - scope[0]) * rng.random())
  def func(data):
    return [[rand() if rng.random() < rate else data[i][j] for j in range(len(data[0]))] for i in range(len(data))]
  return func

# eval
def SSE(x, y):
  e = np.array(x) - np.array(y)
  return np.sum(e * e) / 2

def SSE_diff(x, y):
  e = np.array(x) - np.array(y)
  return e

# init
def He(Nin):
  var = 2 / Nin
  def func(dim):
    return rng.normal(0, var, dim)
  return func

def random(dim):
  return [[rng.random() for _ in range(dim[1])] for _ in range(dim[0])]