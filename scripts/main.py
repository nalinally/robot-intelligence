import network
import layer
from function import *
import matplotlib.pyplot as plt

def accuracy_test(net, inputs, teachers):
  data_size = len(inputs)
  true_or_false = []
  eval_sum = 0
  for i in range(data_size):
    res = net.forward(inputs[i])
    true_or_false.append(list(res).index(np.max(res)) == list(teachers[i]).index(np.max(teachers[i])))
    eval_sum += net.eval_func(teachers[i], res)
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

def main():
  eta = 0.01
  input_dim = 2
  output_dim = 2
  init_weight = 0.5
  output_th = 0.5
  epochs = 1000
  simple_logic_network = network.Network(SSE, SSE_diff, eta)
  simple_logic_network.add_layer(layer.Layer(input_dim, output_dim, ReLU, ReLU_diff, "media", init_weight))
  simple_logic_network.add_layer(layer.Layer(input_dim, output_dim, ReLU, ReLU_diff, "output", init_weight))

  simple_logic_network = learn(simple_logic_network, [[0, 0], [0, 1], [1, 0], [1, 1]], [[1, 0], [1, 0], [1, 0], [0, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]], [[1, 0], [1, 0], [1, 0], [0, 1]], epochs)


if __name__=="__main__":
  main()




