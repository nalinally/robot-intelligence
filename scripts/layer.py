import numpy as np
from function import *


class Layer():

  def __init__(self, input_dim, output_dim, name):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.name = name
    self.input_vec = np.zeros(input_dim)
  
  def f(self, input_vec):
    return np.zeros(self.output_dim)
  
  def df(self, input_vec):
    return np.zeros(self.output_dim, self.input_dim)

  def forward(self, input_vec):
    self.input_vec = input_vec
    return self.f(input_vec)
  
  def back(self, delta_out_vec):
    # print(f"{self.name} back {delta_out_vec}")
    # if self.name=="linear[3]":
      # print(f"{self.name} {self.input_vec} {self.df(self.input_vec)} {delta_out_vec} {self.df(self.input_vec).T @ delta_out_vec}")
      # print(f"// {self.input_vec} {self.df(self.input_vec)} {delta_out_vec} {delta_in_vec} //")
    return self.df(self.input_vec).T @ delta_out_vec
  
class LinearLayer(Layer):

  def __init__(self, input_dim, output_dim, name, eta=0.01, init_func=np.ones):
    super().__init__(input_dim, output_dim, name)
    self.eta = eta
    self.weight = init_func((output_dim, 1 + input_dim))
  
  def f(self, input_vec):
    return self.weight @ np.hstack([[1], input_vec]).T
  
  def df(self, input_vec):
    return self.weight[:, 1:]

  def back(self, delta_out_vec):
    weight_diff = delta_out_vec[:, None] @ np.hstack([[1], self.input_vec])[None, :]
    self.weight = self.weight - (self.eta * weight_diff)
    return super().back(delta_out_vec)
  

class FixedNormalizeLayer(Layer):

  def __init__(self, input_dim, output_dim, name):
    super().__init__(input_dim, output_dim, name)
    self.f = fixed_normalize
    self.df = fixed_normalize_diff
  

class ReLULayer(Layer):
  
  def __init__(self, input_dim, output_dim, name):
    super().__init__(input_dim, output_dim, name)
    self.f = ReLU
    self.df = ReLU_diff
  

class LeakyReLULayer(Layer):

  def __init__(self, input_dim, output_dim, name, a):
    super().__init__(input_dim, output_dim, name)
    self.f = LeakyReLU(a)
    self.df = LeakyReLU_diff(a)


class softMaxLayer(Layer):

  def __init__(self, input_dim, output_dim, name, a):
    super().__init__(input_dim, output_dim, name)
    self.f = softMax(a)
    self.df = softMax_diff(a)


class LinearNormalizeActivateLayer(Layer):

  def __init__(self, input_dim, output_dim, acti_func, acti_diff_func, name, eta=0.01, init_func=np.ones):
    super().__init__(input_dim, output_dim, name)
    self.acti_func = acti_func
    self.acti_diff_func = acti_diff_func
    self.input_vec = np.zeros(input_dim+1)
    self.linearcomb_vec = np.zeros(output_dim)
    self.normalized_vec = np.zeros(output_dim)
    self.mean = 0
    self.std = 0
    self.eta = eta
    self.weight = init_func((output_dim, 1 + input_dim))
  
  def forward(self, input_vec):
    self.input_vec = np.hstack([[1], input_vec]).T
    self.linearcomb_vec = self.weight @ self.input_vec
    self.mean = np.mean(self.linearcomb_vec)
    self.std = np.std(self.linearcomb_vec)
    self.normalized_vec = (self.linearcomb_vec - self.mean) / self.std
    return self.acti_func(self.normalized_vec)
  
  def back(self, delta_out_vec):
    acti_diff_vec = self.acti_diff_func(self.normalized_vec)
    n = float(self.output_dim)
    jacobian = (-1 / (n * self.std)) + (np.eye(self.output_dim) * (1 / self.std)) - (((self.linearcomb_vec[:, None] @ self.linearcomb_vec[None, :]) + np.pow(self.mean, 2) - (self.mean * (self.linearcomb_vec))[:, None] - (self.mean * (self.linearcomb_vec))[None, :]) / (n * np.pow(self.std, 3)))
    weight_diff = (jacobian @ np.array([delta_out_vec * acti_diff_vec]).T) @ np.array([self.input_vec])
    delta_in_vec = ((self.weight.T @ jacobian) @ (delta_out_vec * acti_diff_vec))[1:]
    self.weight = self.weight - (self.eta * weight_diff)
    # print(f"{self.name}: {np.mean(np.abs(weight_diff))}, {np.mean(np.abs(jacobian))}, {np.mean(np.abs(delta_out_vec))}, {np.mean(np.abs(acti_diff_vec))}, {np.mean(np.abs(self.input_vec))}, {np.mean(np.abs(self.weight))}, {np.mean(np.abs(delta_in_vec))}")
    return delta_in_vec
