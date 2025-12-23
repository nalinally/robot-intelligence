import numpy as np

class Layer:

  def __init__(self, input_dim, output_dim, acti_func, acti_diff_func, name, init_weight=0):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.acti_func = acti_func
    self.acti_diff_func = acti_diff_func
    self.name = name
    self.weight = np.ones(output_dim, 1 + input_dim) * init_weight
  
  def forward(self, input_vec):
    return self.weight @ np.vstack([1], input_vec)
  
  def back(self, input_vec, output_vec, delta_out_vec, eta):
    acti_diff_vec = self.acti_diff_func(output_vec)
    weight_diff = (delta_out_vec * acti_diff_vec).T @ input_vec
    delta_in_vec = self.weight.T @ (delta_out_vec * acti_diff_vec)
    self.weight -= eta * weight_diff
    return delta_in_vec

    