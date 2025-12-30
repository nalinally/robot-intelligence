import numpy as np

class Layer:

  def __init__(self, input_dim, output_dim, acti_func, acti_diff_func, name, init_weight=0):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.acti_func = acti_func
    self.acti_diff_func = acti_diff_func
    self.name = name
    self.input_vec = np.zeros(input_dim+1)
    self.linearcomb_vec = np.zeros(output_dim)
    self.weight = np.ones((output_dim, 1 + input_dim)) * init_weight
  
  def forward(self, input_vec):
    self.input_vec = np.hstack([[1], input_vec]).T
    self.linearcomb_vec = self.weight @ self.input_vec
    return self.acti_func(self.linearcomb_vec)
  
  def back(self, delta_out_vec, eta):
    acti_diff_vec = self.acti_diff_func(self.linearcomb_vec)
    weight_diff = np.array([delta_out_vec * acti_diff_vec]).T @ np.array([self.input_vec])
    delta_in_vec = (self.weight.T @ (delta_out_vec * acti_diff_vec))[1:]
    self.weight -= eta * weight_diff
    return delta_in_vec

    