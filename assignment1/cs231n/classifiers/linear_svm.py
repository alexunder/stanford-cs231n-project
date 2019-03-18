import numpy as np
from random import shuffle

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
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        count = 0
        for j in range(num_classes):

            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if j == y[i]:
                continue

            if margin > 0:
                loss += margin
                count += 1
                dW[:, j] = dW[:, j] + X[i]

        dW[:, y[i]] = dW[:, y[i]] - count * X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
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
    num_train = X.shape[0]
    #print("num_train=" + str(num_train))
    scores = np.dot(X, W)
    scores_y = scores[np.arange(scores.shape[0]), y]
    scores_y = scores_y.reshape(-1, 1)
    margins = np.maximum(0, scores - scores_y + 1.0)
    
    # For derivative calculate
    binary_scores = (scores - scores_y + 1.0 > 0).astype(float)
    sum_one = np.sum(binary_scores, axis = 1)
    binary_scores[np.arange(binary_scores.shape[0]), y] = - sum_one
    dW = np.dot(X.T, binary_scores)

    margins[np.arange(margins.shape[0]), y] = 0
    loss = np.sum(margins)
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
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

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

def svm_loss_vectorized_i(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_train = X.shape[0]
    
    for i in range(num_train):
          scores = np.dot(X[i], W)
          # compute the margins for all classes in one vector operation
          margins = np.maximum(0, scores - scores[y[i]] + 1.0)
          # on y-th position scores[y] - scores[y] canceled and gave delta. We want
          # to ignore the y-th position and only consider margin on max wrong class
          margins[y[i]] = 0
          loss_i = np.sum(margins)
          loss += loss_i

    print("loss=" + str(loss))
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return loss, dW