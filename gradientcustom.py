class Stump:
    def __init__(self):
        self.threshold = None #split data
        self.left_value = None #x < threshold
        self.right_value = None #x >= threshold

    def fit(self, X, residuals):
        best_error = float('inf') #tries all thresholds from the training set and chooses the one that gives the smallest squared error.
        for i in range(len(X)): #test x values for split point
            threshold = X[i][0]
            left = [residuals[j] for j in range(len(X)) if X[j][0] < threshold]
            right = [residuals[j] for j in range(len(X)) if X[j][0] >= threshold]

            if not left or not right:
                continue

            left_mean = sum(left) / len(left)
            right_mean = sum(right) / len(right)
#minimize squared error
            error = sum((residuals[j] - (left_mean if X[j][0] < threshold else right_mean))**2 for j in range(len(X)))

            if error < best_error:
                best_error = error
                self.threshold = threshold
                self.left_value = left_mean
                self.right_value = right_mean

    def predict(self, X):
        return [self.left_value if x[0] < self.threshold else self.right_value for x in X]


class GBM:
    def __init__(self, n_estimators=3, learning_rate=0.1): #n_estimator, no of rounds
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_learners = [] #trained stumps
        self.init_prediction = 0

    def fit(self, X, y):
        N = len(y)
        self.init_prediction = sum(y) / N
        current_preds = [self.init_prediction] * N

        for m in range(self.n_estimators):
            # Step (a) Compute residuals
            residuals = [y[i] - current_preds[i] for i in range(N)]

            # Step (b) Fit regression stump to residuals
            stump = Stump()
            stump.fit(X, residuals)

            # Step (c & d) Predict and update model
            predictions = stump.predict(X)
            current_preds = [
                current_preds[i] + self.learning_rate * predictions[i]
                for i in range(N)
            ]

            self.base_learners.append(stump)

    def predict(self, X):
        N = len(X)
        preds = [self.init_prediction] * N
        for stump in self.base_learners:
            stump_preds = stump.predict(X)
            preds = [preds[i] + self.learning_rate * stump_preds[i] for i in range(N)]
        return preds

X = [[1], [2], [3], [4], [5]]
y = [1, 2, 3, 4, 5]

model = GBM(n_estimators=10, learning_rate=0.5)
model.fit(X, y)
predictions = model.predict(X)

print("Predictions:", predictions)