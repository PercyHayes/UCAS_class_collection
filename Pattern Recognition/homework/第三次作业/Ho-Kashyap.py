import numpy as np


Y1 = np.array([[0.1, 1.1, 1],[6.8, 7.1, 1],[-3.5, -4.1, 1],[2.0, 2.7, 1],[4.1, 2.8, 1],[3.1, 5.0, 1],[-0.8, -1.3, 1],[0.9, 1.2, 1],[5.0, 6.4, 1],[3.9, 4.0, 1],
               [3.0, 2.9, -1],[-0.5, -8.7, -1],[-2.9, -2.1, -1],[0.1, -5.2, -1],[4.0, -2.2, -1],[1.3, -3.7, -1],[3.4, -6.2, -1],[4.1, -3.4, -1],[5.1, -1.6, -1],[-1.9, -5.1, -1]]).astype(np.float32)

Y2 = np.array([[7.1, 4.2, 1],[-1.4, -4.3, 1],[4.5, 0.0, 1],[6.3, 1.6, 1],[4.2, 1.9, 1],[1.4, -3.2, 1],[2.4, -4.0, 1],[2.5, -6.1, 1],[8.4, 3.7, 1],[4.1, -2.2, 1],
               [2.0, 8.4, -1],[8.9, -0.2, -1],[4.2, 7.7, -1],[8.5, 3.2, -1],[6.7, 4.0, -1],[0.5, 9.2, -1],[5.3, 6.7, -1],[8.7, 6.4, -1],[7.1, 9.7, -1],[8.0, 6.3, -1]])


class HoKashyap:
    def __init__(self, Y,learning_rate = 0.01, max_iter = 150000,initial_weight = np.array([0,0,0]),initial_error = 1,initial_b = 0.01,bmin = 0.001):
        self.Y = Y
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weight = initial_weight.astype(np.float32)
        self.b = (np.ones(len(Y))*initial_b).astype(np.float32)
        self.bmin = (np.ones(len(Y))*bmin).astype(np.float32)
        self.error = initial_error*np.ones(len(Y)).astype(np.float32)
        self.iteration = 0
        
    
    def train(self):
        while self.iteration < self.max_iter:
            self.error = np.dot(self.Y,self.weight)-self.b
            e_plus = 1/2*(self.error+np.abs(self.error))
            self.b = self.b + 2*self.learning_rate*e_plus
            inv = np.linalg.inv(np.matmul(np.transpose(self.Y),self.Y))
            self.weight = np.matmul(np.matmul(inv,np.transpose(self.Y)),self.b)
            self.iteration += 1
            if np.sum(np.abs(self.error) <= self.bmin)==len(self.error):
                print("Converged")
                print("Iteration: ",self.iteration)
                print("Weight: ",self.weight)
                print("b: ",self.b)
                print("Error: ",self.error)
                return
        print("Did not converge")
        print("Iteration: ",self.iteration)
        print("Weight: ",self.weight)
        print("b: ",self.b)
        return



if __name__ == "__main__":
    print("trainning sample 1")
    hk1 = HoKashyap(Y1)
    hk1.train()
    print("===========================")
    print("trainning sample 2")
    hk2 = HoKashyap(Y2)
    hk2.train()

'''
trainning sample 1
Did not converge
Iteration:  150000
Weight:  [ 0.00345364 -0.00251493  0.00481428]
b:  [0.01       0.01044306 0.01       0.01       0.0119324  0.01
 0.01       0.01       0.01       0.01       0.01       0.01533873
 0.01       0.01       0.01453311 0.01       0.02252061 0.01789637
 0.01682314 0.01      ]
===========================
trainning sample 2
Converged
Iteration:  23076
Weight:  [0.00533945 0.00474587 0.03688261]
b:  [0.0946885  0.01       0.06088606 0.07808403 0.06829845 0.02915888
 0.03070118 0.02127248 0.09925526 0.0483142  0.0136576  0.01
 0.02207932 0.02368253 0.01787022 0.01       0.02320656 0.03993091
 0.04704616 0.03572025]
Error:  [ 3.68655081e-05 -9.99966127e-04  2.40746039e-05  3.05126263e-05
  2.70066001e-05  1.21653600e-05  1.26201931e-05  8.93178408e-06
  3.84549603e-05  1.92345476e-05  4.02088753e-06 -3.10711072e-04
  6.98088941e-06  6.97730222e-06  4.97558740e-06 -5.50908019e-04
  7.25939036e-06  1.32790995e-05  1.62852590e-05  1.17407318e-05]
'''          

        
