import network
import layer
from function import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv
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

def learn(net, inputs_learn, teachers_learn, inputs_test, teachers_test, figname):
  abort_miss_epoch_count = 5
  data_size = len(inputs_learn)
  accuracys = [accuracy(predict(net, inputs_test), teachers_test)]
  evals = [eval(predict(net, inputs_test), teachers_test, net.eval_func)]
  aps = [ap(predict(net, inputs_test), teachers_test)]
  print(f"before: accuracy:{accuracys[-1]} eval:{evals[-1]} ap:{aps[-1]}")
  miss_epoch_count = 0
  abort_flag = False
  epoch = 0
  while not abort_flag:    
    for j in range(data_size):
      net.forward(inputs_learn[j])
      net.back(teachers_learn[j])
    outputs_test = predict(net, inputs_test)
    accuracy_ = accuracy(outputs_test, teachers_test)
    ap_ = ap(outputs_test, teachers_test)
    eval_ = eval(outputs_test, teachers_test, net.eval_func)
    print(f"epoch:{epoch+1} accuracy:{accuracy_} eval:{eval_} ap:{ap_}")
    if len(evals) >= 1:
      miss_epoch_count = 0 if eval_ < np.min(evals) else miss_epoch_count + 1
    if miss_epoch_count >= abort_miss_epoch_count:
      abort_flag = True
    accuracys.append(accuracy_)
    aps.append(ap_)
    evals.append(eval_)
    epoch += 1
  if figname != "":
    plt.figure()
    plt.plot([i for i in range(epoch+1)], accuracys)
    plt.title(f"epoch vs accuracy ({figname})")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig(f"../figures/{figname}_accuracy.png")
    # plt.show(block=False)
    plt.figure()
    plt.plot([i for i in range(epoch+1)], aps)
    plt.title(f"epoch vs AP ({figname})")
    plt.xlabel("epoch")
    plt.ylabel("AP")
    plt.savefig(f"../figures/{figname}_AP.png")
    # plt.show(block=False)
    plt.figure()
    plt.plot([i for i in range(epoch+1)], evals)
    plt.title(f"epoch vs eval ({figname})")
    plt.xlabel("epoch")
    plt.ylabel("eval")
    plt.savefig(f"../figures/{figname}_eval.png")
    # plt.show()

    plt.clf()
    plt.close()

    with open(f"{figname}.csv", "w") as f:
      writer = csv.writer(f)
      writer.writerow(np.hstack([f"{figname}_epoch", [i for i in range(epoch+1)]]))
      writer.writerow(np.hstack([f"{figname}_accuracy", accuracys]))
      writer.writerow(np.hstack([f"{figname}_AP", aps]))
      writer.writerow(np.hstack([f"{figname}_eval", evals]))

  return net, [np.max(accuracys), np.max(aps), np.min(evals)]

def encode_mnist_y(y):
  return [[1 if index==y_ else 0 for index in range(10)] for y_ in y]

def demo_logic():
  eta = 0.01
  input_dim = 2
  output_dim = 2
  simple_logic_network = network.Network(SSE, SSE_diff, eta)
  simple_logic_network.add_layer(layer.Layer(input_dim, output_dim, ReLU, ReLU_diff, "media"))
  simple_logic_network.add_layer(layer.Layer(input_dim, output_dim, ReLU, ReLU_diff, "output"))

  simple_logic_network = learn(simple_logic_network, [[0, 0], [0, 1], [1, 0], [1, 1]], [[1, 0], [1, 0], [1, 0], [0, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]], [[1, 0], [1, 0], [1, 0], [0, 1]], "")

def noise_test(net, x_train, y_train, x_test_, y_test, noise_func, prefix):
  train_data_size = len(x_train)
  test_data_size = len(x_test_)
  x_train_1d = np.array(x_train).reshape([train_data_size, -1])
  y_train_vec = encode_mnist_y(y_train)
  y_test_vec = encode_mnist_y(y_test)
  # noises = np.linspace(0, 0.25, 10)
  noises = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
  accuracys = []
  aps = []
  evals = []
  for noise_rate in noises:
    print(f"noise:{noise_rate}")
    noise = noise_func(noise_rate)
    x_test = [noise(x_test__) for x_test__ in x_test_]
    x_test_1d = np.array(x_test).reshape([test_data_size, -1])
    net.init_layers()
    _, res = learn(net, x_train_1d, y_train_vec, x_test_1d, y_test_vec, f"{prefix}_nse{noise_rate}")
    accuracys.append(res[0])
    aps.append(res[1])
    evals.append(res[2])
  plt.figure()
  plt.plot(noises, accuracys)
  plt.title(f"noise rate vs accuracy ({prefix})")
  plt.xlabel("noise rate")
  plt.ylabel("accuracy")
  plt.savefig(f"../figures/{prefix}_accuracy.png")
  plt.figure()
  plt.plot(noises, aps)
  plt.title(f"noise rate vs AP ({prefix})")
  plt.xlabel("noise rate")
  plt.ylabel("AP")
  plt.savefig(f"../figures/{prefix}_AP.png")
  plt.figure()
  plt.plot(noises, evals)
  plt.title(f"noise rate vs eval ({prefix})")
  plt.xlabel("noise rate")
  plt.ylabel("eval")
  plt.savefig(f"../figures/{prefix}_eval.png")

  plt.clf()
  plt.close()

  with open(f"{prefix}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(np.hstack([f"{prefix}_accuracy", accuracys]))
    writer.writerow(np.hstack([f"{prefix}_AP", aps]))
    writer.writerow(np.hstack([f"{prefix}_eval", evals]))

def datasize_noise_test(net, x_train_, y_train_, x_test, y_test, noise_func):
  down_sampling_rates = [5, 50, 500]
  for down_sampling_rate in down_sampling_rates:
    x_train = x_train_[::down_sampling_rate]
    y_train = y_train_[::down_sampling_rate]
    net.init_layers()
    noise_test(net, x_train, y_train, x_test, y_test, noise_func, f"dsp{down_sampling_rate}")

def main():
  leaky_relu_a = 0.01
  softmax_a = 1
  eta = 0.01
  # down_sampling_rate_train = 50
  down_sampling_rate_test = 5
  pooling_rate = 2

  (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
  # pooling = index_pooling(pooling_rate, pooling_rate)
  pooling = average_pooling(pooling_rate, pooling_rate)
  noise_func = lambda x: white_noise(x, [0, 1])
  normalize = data_normalize([0, 1], [0, 255])
  x_train = [normalize(pooling(x_train_)) for x_train_ in x_train_all]
  y_train = y_train_all
  x_test = [normalize(pooling(x_test_)) for x_test_ in x_test_all[::down_sampling_rate_test]]
  y_test = y_test_all[::down_sampling_rate_test]
  train_data_size = len(x_train)
  # test_data_size = len(x_test)
  x_train_1d = np.array(x_train).reshape([train_data_size, -1])
  # x_test_1d = np.array(x_test).reshape([test_data_size, -1])
  # y_train_vec = encode_mnist_y(y_train)
  # y_test_vec = encode_mnist_y(y_test)
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
  # number_id_network = learn(number_id_network, x_train_1d, y_train_vec, x_test_1d, y_test_vec, "")
  # noise_test(number_id_network, x_train, y_train, x_test, y_test, noise_func, "")
  datasize_noise_test(number_id_network, x_train, y_train, x_test, y_test, noise_func)

if __name__=="__main__":
  main()




