# Activation Functions

- Applying non-linear transformations to activate or deactivate a neuron
- Step Function
    - For some $\theta$ (the threshold value),
        
        $f(x) = \Biggl\{ \begin{matrix} 1  ~ ~ ~ if ~ ~ ~ x \ge \theta \\ 0 ~ ~ ~ otherwise\end{matrix}$
        
    - Code:
        
        ```python
        def step(self):
            thresh = self.params['threshold']
            self.data = np.where(self.data >= thresh, 1, 0)
        ```
        
- Sigmoid Function
    - To get an output between 0 and 1 (both excluded),
        
        $f(x) = \frac{1}{1 + e ^ {-x}}$
        
    - Used in binary-classification problems
    - Code:
        
        ```python
        def sigmoid(self):
            self.data = 1 / (1 + np.exp(-self.data))
        ```
        
- TanH Function
    - To get the output between -1 and 1 (both excluded),
        
        $f(x) = \frac{2}{1 + e ^ {-2x}} - 1$
        
    - Code:
        
        ```python
        def tanh(self):
            self.data = np.tanh(self.data)
        ```
        
- ReLU Function
    - General activation function used
        
        $f(x) = max(0, x)$
        
    - Dying ReLU problem: After many iterations, a dead state in a neuron can occur where the output is only 0 so weights can not be updated
    - Code:
        
        ```python
        def relu(self):
            self.data = np.where(self.data >= 0, self.data, 0)
        ```
        
- Leaky ReLU Function
    - To avoid the dying ReLU problem,
        
        $f(x) = \Biggl\{ \begin{matrix} x  ~ ~ ~ if ~ ~ ~ x \ge 0 \\ a \cdot x ~ ~ ~ otherwise\end{matrix}$
        
    - The value of ‘a’ (also known as the scaling factor): very small but not 0
    - Code:
        
        ```python
        def leaky_relu(self):
            alpha = self.params['a']
            self.data = np.where(self.data >= 0, self.data, alpha * self.data)
        ```
        
- Softmax Function
    - Output is between 0 and 1: probability values
    - The raw number ($e^{y_{i}}$) $\propto$ probability
        
        $S(y_i) = \frac{e^{y_i}}{\sum{e ^ {y_i}}}$
        
    - Used in multiclass classification problems
    - Code:
        
        ```python
        def softmax(self):
            self.data = np.exp(self.data) / np.sum(np.exp(self.data), axis = 0)
        ```