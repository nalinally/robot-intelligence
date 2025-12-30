import network
import layer
from function import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
rng = np.random.default_rng()

def accuracy_test(net, inputs, teachers):
  data_size = len(inputs)
  true_or_false = []
  eval_sum = 0
  for i in range(data_size):
    res = net.forward(inputs[i])
    # true_or_false.append(rng.integers(10) == list(teachers[i]).index(np.max(teachers[i])))
    true_or_false.append(list(res).index(np.max(res)) == list(teachers[i]).index(np.max(teachers[i])))
    eval_sum += net.eval_func(teachers[i], res)
    print(f"{list(res).index(np.max(res))}--{list(teachers[i]).index(np.max(teachers[i]))} {true_or_false[-1]}")
  print(f"{true_or_false.count(True)}/{data_size}")
  return float(true_or_false.count(True)) / float(data_size)

def eval(net, inputs, teachers):
  data_size = len(inputs)
  eval_sum = 0
  for i in range(data_size):
    res = net.forward(inputs[i])
    eval_sum += net.eval_func(teachers[i], res)
  return float(eval_sum) / float(data_size)

def learn(net, inputs_learn, teachers_learn, inputs_test, teachers_test, epoch):
  data_size = len(inputs_learn)
  accuracys = []
  evals = []
  for i in range(epoch):    
    for j in range(data_size):
      net.forward(inputs_learn[j])
      net.back(teachers_learn[j])
    accuracy = accuracy_test(net, inputs_test, teachers_test)
    eval_ = eval(net, inputs_test, teachers_test)
    print(f"epoch:{i} accuracy:{accuracy} eval:{eval_}")
    accuracys.append(accuracy)
    evals.append(eval_)
  plt.plot([i+1 for i in range(epoch)], accuracys, label="epoch vs accuracy")
  plt.plot([i+1 for i in range(epoch)], evals, label="epoch vs eval")
  plt.show()
  return net

def encode_mnist_y(y):
  return [[1 if index==y_ else 0 for index in range(10)] for y_ in y]

def demo_logic():
  eta = 0.01
  input_dim = 2
  output_dim = 2
  epoch = 500
  simple_logic_network = network.Network(SSE, SSE_diff, eta)
  simple_logic_network.add_layer(layer.Layer(input_dim, output_dim, ReLU, ReLU_diff, "media"))
  simple_logic_network.add_layer(layer.Layer(input_dim, output_dim, ReLU, ReLU_diff, "output"))

  simple_logic_network = learn(simple_logic_network, [[0, 0], [0, 1], [1, 0], [1, 1]], [[1, 0], [1, 0], [1, 0], [0, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]], [[1, 0], [1, 0], [1, 0], [0, 1]], epoch)

def main():
  eta = 0.01
  epoch = 100
  down_sampling_rate = 50

  (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
  x_train = x_train_all[::down_sampling_rate]
  y_train = y_train_all[::down_sampling_rate]
  x_test = x_test_all[::down_sampling_rate]
  y_test = y_test_all[::down_sampling_rate]
  train_data_size = len(x_train)
  test_data_size = len(x_test)
  x_train_1d = x_train.reshape([train_data_size, -1])
  x_test_1d = x_test.reshape([test_data_size, -1])
  y_train_vec = encode_mnist_y(y_train)
  y_test_vec = encode_mnist_y(y_test)
  image_size = len(x_train_1d[0])

  layer_dims = [image_size, 324, 144, 64, 25, 10]
  number_id_network = network.Network(SSE, SSE_diff, eta)
  for i in range(len(layer_dims)-1):
    number_id_network.add_layer(layer.Layer(layer_dims[i], layer_dims[i+1], ReLU, ReLU_diff, "media", He(layer_dims[i])))

  number_id_network = learn(number_id_network, x_train_1d, y_train_vec, x_test_1d, y_test_vec, epoch)

if __name__=="__main__":
  main()




