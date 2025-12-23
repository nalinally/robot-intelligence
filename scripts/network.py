import numpy as np
import layer
from activate_function import *

class Network:

  def __init__(self, input_dim, eval_func, eval_diff_func,eta):
    self.layers = [layer.Layer(input_dim, input_dim, one, one_diff, "input", 0)]
    self.vecs = []
    self.deltas = []
    self.eval_func = eval_func
    self.eval_diff_func = eval_diff_func
    self.eta = eta
  
  def add_layer(self, layer):
    if self.layers[-1].output_dim == layer.input_dim:
      print(f"add_layer: layer[{self.layers[-1].name}]と挿入しようとしているレイヤーの次元が合いません")
      return
    self.layers.append(layer)

  def forward(self, input_vec):
    self.vecs = [input_vec]
    for layer in self.layers:
      self.vecs.append(layer.forward(self.vecs[-1]))
    return self.vecs[-1]
  
  def back(self, output_vec):
    self.deltas = []
    for i in range(len(self.layers)).reverse():
      self.deltas.insert(0, self.layers[i].back(self.vecs[i], self.vecs[i+1], self.deltas[-1]))



