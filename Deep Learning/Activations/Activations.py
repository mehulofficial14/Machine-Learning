'''
Input Dimensions: Dimensions of X
Output Dimensions: Dimensions of X
'''
class Activation:
  def __init__(self, name, X, params):
    self.f = name
    self.data = data
    self.f_map = {
        'step': step,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu,
        'leaky-relu': leaky_relu,
        'softmax': softmax
    }
    self.params = params
  def step(self):
    thresh = self.params['threshold']
    self.data = np.where(self.data >= thresh, 1, 0)
  def sigmoid(self):
    self.data = 1 / (1 + np.exp(-self.data))
  def tanh(self):
    self.data = np.tanh(self.data)
  def relu(self):
    self.data = np.where(self.data >= 0, self.data, 0)
  def leaky_relu(self):
    alpha = self.params['a']
    self.data = np.where(self.data >= 0, self.data, alpha * self.data)
  def softmax(self):
    self.data = np.exp(self.data) / np.sum(np.exp(self.data), axis = 0)
