import network
import layer
from function import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
rng = np.random.default_rng()

def predict(net, inputs):
  return [net.forward(input) for input in inputs]

def accuracy(outputs, teachers):
  return np.mean([1 if list(output).index(np.max(output)) == list(teacher).index(np.max(teacher)) else 0 for output, teacher in zip(outputs, teachers)])

def eval(outputs, teachers, f):
  return np.mean([f(output, teacher) for output, teacher in zip(outputs, teachers)])

def auc(indexes):
  # print(indexes)
  return np.mean([float(i+1) / float(indexes[i]+1) for i in range(len(indexes))])

def ap(outputs, teachers):
  # print(f"{np.array(outputs)[:, 5]} {np.argsort(outputs, axis=0)[:, 5]}")
  args = np.argsort(np.argsort(-np.array(outputs), axis=0), axis=0)
  # print(f"{args[:, 5]} {np.array(outputs)[:, 5]} {np.array(teachers)[:, 5]} {np.array(teachers)[:, 5] != 0}")
  return np.mean([auc(np.sort(args[:, i][np.array(teachers)[:, i] != 0])) for i in range(len(outputs[0]))])

def learn(net, inputs_learn, teachers_learn, inputs_test, teachers_test, epoch):
  data_size = len(inputs_learn)
  accuracys = [accuracy(predict(net, inputs_test), teachers_test)]
  evals = [eval(predict(net, inputs_test), teachers_test, net.eval_func)]
  aps = [ap(predict(net, inputs_test), teachers_test)]
  print(f"before: accuracy:{accuracys[-1]} eval:{evals[-1]} ap:{aps[-1]}")
  for i in range(epoch):    
    for j in range(data_size):
      net.forward(inputs_learn[j])
      net.back(teachers_learn[j])
    outputs_test = predict(net, inputs_test)
    accuracy_ = accuracy(outputs_test, teachers_test)
    ap_ = ap(outputs_test, teachers_test)
    eval_ = eval(outputs_test, teachers_test, net.eval_func)
    print(f"epoch:{i+1} accuracy:{accuracy_} eval:{eval_} ap:{ap_}")
    accuracys.append(accuracy_)
    aps.append(ap_)
    evals.append(eval_)
  plt.figure()
  plt.plot([i for i in range(epoch+1)], accuracys)
  plt.title("epoch vs accuracy")
  plt.show(block=False)
  plt.figure()
  plt.plot([i for i in range(epoch+1)], aps)
  plt.title("epoch vs aps")
  plt.show(block=False)
  plt.figure()
  plt.plot([i for i in range(epoch+1)], evals)
  plt.title("epoch vs evals")
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
  leaky_relu_a = 0.01
  softmax_a = 1
  eta = 0.01
  epoch = 50
  down_sampling_rate_train = 50
  down_sampling_rate_test = 5
  pooling_rate = 2
  noise_rate = 0.01

  (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
  # pooling = index_pooling(pooling_rate, pooling_rate)
  pooling = average_pooling(pooling_rate, pooling_rate)
  noise = white_noise(noise_rate, [0, 255])
  x_train = [noise(pooling(x_train_)) for x_train_ in x_train_all[::down_sampling_rate_train]]
  y_train = y_train_all[::down_sampling_rate_train]
  x_test = [noise(pooling(x_test_)) for x_test_ in x_test_all[::down_sampling_rate_test]]
  y_test = y_test_all[::down_sampling_rate_test]
  train_data_size = len(x_train)
  test_data_size = len(x_test)
  x_train_1d = np.array(x_train).reshape([train_data_size, -1])
  x_test_1d = np.array(x_test).reshape([test_data_size, -1])
  y_train_vec = encode_mnist_y(y_train)
  y_test_vec = encode_mnist_y(y_test)
  image_size = len(x_train_1d[0])

  layer_dims = [image_size, 49, 25, 10]
  number_id_network = network.Network(SSE, SSE_diff, eta)
  number_id_network.add_layer(layer.FixedNormalizeLayer(layer_dims[0], layer_dims[0], "input_normalize"))
  for i in range(len(layer_dims)-2):
    number_id_network.add_layer(layer.LinearLayer(layer_dims[i], layer_dims[i+1], f"linear[{i}]", eta, He(layer_dims[i])))
    number_id_network.add_layer(layer.FixedNormalizeLayer(layer_dims[i+1], layer_dims[i+1], f"normalize[{i}]"))
    number_id_network.add_layer(layer.LeakyReLULayer(layer_dims[i+1], layer_dims[i+1], f"leakyrelu[{i}]", leaky_relu_a))
  number_id_network.add_layer(layer.LinearLayer(layer_dims[-2], layer_dims[-1], f"linear[{len(layer_dims)-2}]", eta, He(layer_dims[-2])))
  number_id_network.add_layer(layer.FixedNormalizeLayer(layer_dims[-1], layer_dims[-1], f"normalize[{len(layer_dims)-2}]"))
  number_id_network.add_layer(layer.softMaxLayer(layer_dims[-1], layer_dims[-1], f"softmax[{len(layer_dims)-2}]", softmax_a))

  number_id_network.show()
  number_id_network = learn(number_id_network, x_train_1d, y_train_vec, x_test_1d, y_test_vec, epoch)

if __name__=="__main__":
  main()




