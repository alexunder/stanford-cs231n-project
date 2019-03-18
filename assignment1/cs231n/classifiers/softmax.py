import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
      scores = np.dot(X[i], W)
      scores -= np.max(scores)
      sum_scores = np.sum(np.exp(scores))
      loss_i = np.exp(scores[y[i]]) / sum_scores
      loss += - np.log(loss_i)

      for j in range(num_class):
          if j == y[i]:
              dW[:, j] += (np.exp(scores[j]) / sum_scores - 1) * X[i]
          else:
              dW[:, j] += (np.exp(scores[j]) / sum_scores) * X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW +=2 * reg * W
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
  num_train = X.shape[0]
  scores = np.dot(X, W)
  scores = scores - np.max(scores, axis = 1).reshape(-1, 1)
  #print("dbg2 scores 0=" + str(scores[0]))
  sum_scores = np.sum(np.exp(scores), axis = 1)
  scores_y = np.exp(scores[np.arange(scores.shape[0]), y])
  loss_array = -np.log(scores_y / sum_scores)
  loss = np.sum(loss_array)

  #For dW
  scores_dw_buffer = np.exp(scores) / sum_scores.reshape(-1, 1)
  scores_dw_buffer[np.arange(scores_dw_buffer.shape[0]), y] -= 1
  dW = np.dot(X.T, scores_dw_buffer) 

  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW +=2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

