import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  for i in range(N):
      score = np.dot(X[i,:], W)
      unnormalized_prob = np.exp(score)
      sum_prob = np.sum(unnormalized_prob)
      prob = np.zeros(shape = (C,1))          
      for j in range(C):
          prob[j] = unnormalized_prob[j]/sum_prob
      loss += -np.log(prob[y[i]])
      
      for k in range(C):
          if k == y[i]:
              dW[:,k] += -X[i] + unnormalized_prob[k]/sum_prob * X[i]
          if k != y[i]:
              dW[:,k] += unnormalized_prob[k]/sum_prob * X[i]
  
  loss /= N
  dW /= N
  loss += reg*np.sum(W*W)
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  scores = np.dot(X, W)
  unnormalized_scores = np.exp(scores)
  sum_prob = np.sum(unnormalized_scores, axis = 1)
  sum_matrix = np.tile(sum_prob, (C,1))
  prob = unnormalized_scores / np.transpose(sum_matrix)
  loss += -np.sum(np.log(prob[range(N), y]))
  
  right_points = prob
  right_points[range(N), y] -= 1   #to realize the minus 1 in naive function

  dW = np.dot(np.transpose(X), right_points)
  
  loss /= N
  dW /= N
  loss += reg*np.sum(W*W)
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

