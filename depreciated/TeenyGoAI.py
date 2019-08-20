import numpy as np

# define neural network model
class NeuralNetwork(object):

  def __init__(self, x, y, alpha=0.12, iterations=1000, num_layers=5, hidden_addition=2, lamb=0.1):

    # initiate class properties
    self.x = x
    self.y = y
    self.alpha = alpha
    self.lamb = lamb
    self.iterations = iterations
    self.num_layers = num_layers
    self.w = {}
    self.b = {}
    self.z = {}
    self.a = {}
    self.e = {}
    self.historical_cost = []
    self.hidden_addition = hidden_addition

    # create layer weights and layer variables
    for i in range(1, self.num_layers+hidden_addition):

      # if not the last weight initiate with eacher layer one bigger than the input size of x


      if i != self.num_layers:
        if i == 1:
          self.w["w"+str(i)] = np.random.randn(self.x.shape[1], self.x.shape[1]+hidden_addition)*0.01
          self.b["b"+str(i)] = np.random.randn(1, self.x.shape[1]+hidden_addition)*0.01
        else:
          self.w["w"+str(i)] = np.random.randn(self.x.shape[1]+hidden_addition, self.x.shape[1]+hidden_addition)*0.01
          self.b["b"+str(i)] = np.random.randn(1, self.x.shape[1]+hidden_addition)*0.01

      # if the last weight initiate with output = dimensions of y
      else:
        self.w["w"+str(i)] = np.random.randn(self.x.shape[1]+hidden_addition, self.y.shape[1])*0.01
        self.b["b"+str(i)] = np.random.randn(1, self.y.shape[1])*0.01

      # doesnt matter, will be changed later
      self.z["z" + str(i)] = np.zeros([self.x.shape[0], self.x.shape[1]])
      self.a["a" + str(i)] = np.zeros([self.x.shape[0], self.x.shape[1]])

  # calculate and make predictions
  def forward_propagation(self):

    # initiate forward propigation with dotting x an w1
    self.a['a0'] = self.x
    self.z["z1"] = np.dot(self.x, self.w["w1"]) + self.b['b1']
    self.a['a1'] = np.tanh(self.z['z1'])

    # iterate through all dots and activations
    for i in range(2, self.num_layers):
      self.z["z" + str(i)] = np.dot(self.a['a'+str(i-1)], self.w['w' + str(i)]) + self.b['b' + str(i)]
      self.a['a'+ str(i)] = np.tanh(self.z["z" + str(i)])

    # on the last iteration use sigmoid instead of tanh for classification
    self.z["z" + str(self.num_layers)] = np.dot(self.a['a'+str(self.num_layers-1)], self.w['w' + str(self.num_layers)]) + self.b['b' + str(self.num_layers)]
    self.a['a'+ str(self.num_layers)] = self.sigmoid(self.z["z" + str(self.num_layers)])
    return self.a['a'+str(self.num_layers)]

  # adjust weights based on cost function
  def backward_propagation(self):
    self.create_updates()
    #self.w['w3'] = self.w['w3'] - (self.alpha/self.x.shape[0])*np.dot((self.sigmoid_prime(self.z['z3'])*self.j_prime()).T, self.a['a2']).T

    # iterate throught weights
    for i in reversed(range(1, self.num_layers+1)):
      self.w["w"+str(i)] = self.w["w"+str(i)] - (self.alpha/self.x.shape[0])*np.dot(self.a['a'+str(i-1)].T, self.e['e'+str(i)]) + self.lamb*np.sum(self.w['w'+str(i)])/self.x.shape[0]
      self.b["b"+str(i)] = self.b["b"+str(i)] - (self.alpha/self.x.shape[0])*np.sum(self.e['e'+str(i)], axis=0) + self.lamb*np.sum(self.b['b'+str(i)])/self.x.shape[0]


  # returns the derivative of the cost
  def j_prime(self):
    return (self.y/self.a['a' + str(self.num_layers)] - (1-self.y)/(1-self.a['a' + str(self.num_layers)]))

  # creates update cache
  def create_updates(self):
    self.e['e'+str(self.num_layers)] = self.j_prime()*self.sigmoid_prime(self.z['z'+str(self.num_layers)])
    for i in reversed(range(1, self.num_layers)):
      self.e['e'+str(i)] =  np.dot(self.e['e' + str(i+1)], self.w['w' + str(i+1)].T)*self.tanh_prime(self.z['z'+str(i)])


  # optimize model based on inputes
  def optimize(self):
      self.forward_propagation()
      self.backward_propagation()
      self.historical_cost.append(self.cost_function())

  # calculate the cost of prections
  def cost_function(self):
    j = -(np.sum(self.y*np.log(self.a['a'+str(self.num_layers)]) + (1-self.y)*np.log(1 - self.a['a'+ str(self.num_layers)]))/self.x.shape[0])
    return j

  # sigmoid activation function
  def sigmoid(self, z):
    return 1/(1+np.exp(-z))

  # derivative of sigmoid activation function
  def sigmoid_prime(self, z):
    return -np.exp(-z)/np.square(1 + np.exp(-z))

  # derivative of tanh activation function
  def tanh_prime(self, z):
    return 1 - np.square(np.tanh(z))


  def save_weights(self):
    # save weights
    for k in range(1, self.num_layers+1):
        stringW = ""
        shapeW = self.w["w"+str(k)].shape
        for i in range(shapeW[0]):
            for j in range(shapeW[1]):
                stringW = stringW + ";" + str(self.w["w"+str(k)][i][j])
        stringW = stringW + "\n"
        file = open("5-LayerNeuralParamters/weights" + str(k)+".txt", "w")
        file.write(stringW)

    # save bias
    for k in range(1, self.num_layers+1):
        stringB = ""
        shapeB = self.b["b"+str(k)].shape
        for i in range(shapeB[0]):
            for j in range(shapeB[1]):
                stringB = stringB + ";" + str(self.b["b"+str(k)][i][j])
        stringB = stringB + "\n"
        file = open("5-LayerNeuralParamters/bias" + str(k)+".txt", "w")
        file.write(stringB)








  def load_weights(self):
      pass
