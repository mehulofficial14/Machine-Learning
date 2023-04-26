#Input dimensions and outout dimensions are the same as X
class Evaluate:
  def __init__(self, data):
    self.data = data
  def cross_entropy(self):
    self.data = -np.log(self.data)
