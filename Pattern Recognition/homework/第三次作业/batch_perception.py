
import numpy as np

trainset1 = np.array([[0.1, 1.1, 1],[6.8, 7.1, 1],[-3.5, -4.1, 1],[2.0, 2.7, 1],[4.1, 2.8, 1],[3.1, 5.0, 1],[-0.8, -1.3, 1],[0.9, 1.2, 1],[5.0, 6.4, 1],[3.9, 4.0, 1],
    [-7.1, -4.2, -1],[1.4, 4.3, -1],[-4.5, -0.0, -1],[-6.3, -1.6, -1],[-4.2, -1.9, -1],[-1.4, 3.2, -1],[-2.4, 4.0, -1],[-2.5, 6.1, -1],[-8.4, -3.7, -1],[-4.1, 2.2, -1]]).astype(np.float32)

trainset2 = np.array([[-7.1, -4.2, -1],[1.4, 4.3, -1],[-4.5, -0.0, -1],[-6.3, -1.6, -1],[-4.2, -1.9, -1],[-1.4, 3.2, -1],[-2.4, 4.0, -1],[-2.5, 6.1, -1],[-8.4, -3.7, -1],[-4.1, 2.2, -1],
    [-3.0, -2.9, 1],[0.5, 8.7, 1],[2.9, 2.1, 1],[-0.1, 5.2, 1],[-4.0, 2.2, 1],[-1.3, 3.7, 1],[-3.4, 6.2, 1],[-4.1, 3.4, 1],[-5.1, 1.6, 1],[1.9, 5.1, 1]]).astype(np.float32)

class perceptron:
    def __init__(self, trainset, learning_rate=0.001, max_iterations=1000):
        self.trainset = trainset
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = np.zeros(3).astype(np.float32)
        self.iteration = 0

    def train(self):
        while np.sum(np.sum(np.dot(self.trainset, self.weights) > 0)) != len(self.trainset):
            sample_sum = np.zeros(3).astype(np.float32)
            for x in self.trainset:
                if np.dot(x, self.weights)  <= 0:
                    #self.weights += self.learning_rate * x
                    sample_sum += x
            self.iteration += 1
            self.weights += sample_sum*self.learning_rate
            #print("rec: ", rec*self.learning_rate)

        print("weights: ", self.weights)
        print("iterations: ", self.iteration)
        return self.weights

if __name__ == "__main__":
    p1 = perceptron(trainset1,learning_rate=0.01)
    weights1 = p1.train()
    #weights = np.array([-0.3870,0.4780,0.6610])
    #print("final result: ",np.dot(trainset1, weights1))

    p2 = perceptron(trainset2)
    weights2 = p2.train()
    #print("final result: ",np.dot(trainset2, weights2))


