import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] += -X[i,:]    
        dW[:,j] += X[i,:]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg*2*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  N = X.shape[0]
  C = W.shape[1]
  correct_scores_vector = np.zeros((N,1))
  #for i in range(N):
  #    correct_scores_vector[i] = scores[i][y[i]]
  correct_scores_vector = scores[range(N),y]
  correct_scores_matrix = np.tile(correct_scores_vector,(C, 1))
  intermediate = np.maximum(0, scores - np.transpose(correct_scores_matrix) + 1)
  #for i in range(N):
  #    intermediate[i][y[i]] = 0
  intermediate[range(N), y] = 0
  loss = np.mean(np.sum(intermediate, axis = 1)) + reg * np.sum(W * W)
  
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  flag = intermediate
  flag[intermediate > 0] = 1
  sum_flag = np.sum(flag, axis = 1) #num of wrong in every sample
  #print(sum_flag)
  #for i in range(N):
  #  flag[i][y[i]] -= sum_flag[i] #only modify the right labels when the prediction is wrong, make them into -1;right choice set to 0
  flag[range(N), y] -= sum_flag[range(N)]
  
  #print(flag)
  dW = np.dot(np.transpose(X), flag)
  
  dW /= N
  
  dW += reg*2*W
    
  ##################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
