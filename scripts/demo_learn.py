from main import learn, encode_mnist_y
from function import *
import layer
import network
import tensorflow as tf

def main():
  leaky_relu_a = 0.01
  softmax_a = 1
  eta = 0.01
  down_sampling_rate_train = 5
  down_sampling_rate_test = 5
  pooling_rate = 2

  (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
  pooling = average_pooling(pooling_rate, pooling_rate)
  normalize = data_normalize([0, 1], [0, 255])
  x_train = [normalize(pooling(x_train_)) for x_train_ in x_train_all[::down_sampling_rate_train]]
  y_train = y_train_all[::down_sampling_rate_train]
  x_test = [normalize(pooling(x_test_)) for x_test_ in x_test_all[::down_sampling_rate_test]]
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
  number_id_network = learn(number_id_network, x_train_1d, y_train_vec, x_test_1d, y_test_vec, "demo_learn")

if __name__=="__main__":
  main()