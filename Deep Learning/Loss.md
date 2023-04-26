# Evaluation Metrics

- Classification evaluation metrics
    - Accuracy = $\frac{Correctly ~ classified ~  instances}{Total ~ instances}$
    - Precision = $\frac{Correctly ~ labelled ~ positive ~ instances}{Total ~ positive ~ labelled ~ instances}$ or how many retireved items are relevant
    - Recall = $\frac{Correctly ~ labelled ~ positive ~ instances}{Total ~ instances ~ that ~ are ~ actually ~ positive}$ or how many relevant items are retrieved
    - F1 Score = $\frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$
    - PR curve: precision-recall curve preferring a high precision and a high recall value that is the top-right corner of the graph
    - ROC curve: compare the true-positive and the false-positive rates
    - AUC: The area under the curve value should be high
    - Cross Entropy: difference/distance between two probability distributions
        - Binary Cross Entropy
        - Categorical Cross Entropy
            
            ```python
            def cross_entropy(self):
                self.data = -np.log(self.data)
            ```
            
        - Sparse Categorical Cross Entropy
- Regression evaluation metrics
    - Mean-Absolute Error = $\frac{1}{n}\sum_{j = 1}^{n}{|y_{j} - \hat{y_j}|}$
    - Mean-Squared Error = $\frac{1}{n}\sum_{j = 1}^{n}{(y_{j} - \hat{y_j}) ^ 2}$
    - Root-Mean-Squared Error = $\sqrt{\frac{1}{n}\sum_{j = 1}^{n}{(y_{j} - \hat{y_j}) ^ 2}}$
    - R-Squared (Coefficient of Determination)
        - Tells about how well the model fits the data
        - The value lies between 0 (does not fit the data at all) and 1 (fits the data completely)
    - Cosine similarity