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
      
  for i in range(X.shape[0]):
      f_class = np.exp(np.sum(X[i,:] * W[:,y[i]]))
      f_total = 0
      for j in range(W.shape[1]):                
          f_total = f_total + np.exp(np.sum(X[i,:] * W[:,j]))
      for j in range(W.shape[1]):
          if j != y[i]:
              dW[:, j] = dW[:, j] + (np.exp(np.sum(X[i,:] * W[:,j])) / f_total) * X[i,:]
      dW[:, y[i]] = dW[:, y[i]] + (f_class / f_total - 1) * X[i,:]
      lossi = np.log(f_class / f_total)
      loss = loss - lossi
  loss /= X.shape[0] + reg * np.sum(W * W)
  dW   /= X.shape[0] + 2 * reg * W
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
  num_classes = W.shape[1]

  #loss
  linear = np.exp(X.dot(W)) 
  class_total = np.sum(linear, axis = 1)
  loss  = np.sum(np.log(linear[np.arange(len(linear)),y]/class_total))
  loss = -loss/X.shape[0]

  # gradient
  mat_class = np.reshape(np.repeat(class_total, num_classes, axis = 0), (y.shape[0],num_classes))
  temp = linear/mat_class
  temp[np.arange(temp.shape[0]),y] = temp[np.arange(temp.shape[0]),y]-1
  dW = X.T.dot(temp)
  dW = dW/X.shape[0] + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

