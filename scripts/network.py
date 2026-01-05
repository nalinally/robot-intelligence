import numpy as np
import layer

class Network:

  def __init__(self, eval_func, eval_diff_func, eta):
    self.layers = []
    self.vecs = []
    self.deltas = []
    self.eval_func = eval_func
    self.eval_diff_func = eval_diff_func
    self.eta = eta
  
  def add_layer(self, layer):
    if len(self.layers) != 0:
      if self.layers[-1].output_dim != layer.input_dim:
        print(f"add_layer: layer[{self.layers[-1].name}]と挿入しようとしているレイヤーの次元が合いません")
        return
    self.layers.append(layer)

  def forward(self, input_vec):
    self.vecs = [input_vec]
    for layer in self.layers:
      self.vecs.append(layer.forward(self.vecs[-1]))
    # print(f"{input_vec}, {self.vecs[-1]}")
    return self.vecs[-1]
  
  def back(self, teacher_vec):
    self.deltas = [self.eval_diff_func(self.vecs[-1], teacher_vec)]
    for i in reversed(range(len(self.layers))):
      # print(f"{self.layers[i].name} {self.deltas[0]}")
      # delta = self.layers[i].back(self.deltas[0])
      # print(f"{self.layers[i].name} {delta}")
      self.deltas.insert(0, self.layers[i].back(self.deltas[0]))
      # print(f"{self.layers[i].name} {self.deltas[0]}")
    # print(self.deltas)

  def init_layers(self):
    for layer in self.layers:
      layer.init_layer()

  def eval(self, teacher_vec):
    return self.eval_func(self.vecs[-1], teacher_vec)
  
  def show(self):
    for layer in self.layers:
      print(f"{layer.name} ({layer.input_dim} > {layer.output_dim})")



