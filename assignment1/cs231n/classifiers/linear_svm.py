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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    dwC = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dwC += 1
        dW[:,j] += X[i]  

    dW[:,y[i]] -= dwC*X[i] 
    dW += reg * W
     
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

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

  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) 
  correct_scores = scores[list(range(num_train)), y]  

  # Add an extra dimension, convert shape from (n,) to (n,1)
  correct_scores = np.reshape(correct_scores,(num_train,1))

  # Repeat the same column to get the shape (n,num_classes)
  correct_scores = np.repeat(correct_scores,num_classes,axis=1)

  #Compute the SVM Loss 
  margins = scores - correct_scores + 1
  margins = np.maximum(margins,0)
  loss = np.sum(margins) - num_train
  loss /= num_train 
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
  
  # Convert the margins array to an array of 1's and 0's based on whether margin is greater than 0
  margins_boolean = margins > 0 
  margins_boolean = margins_boolean.astype(int)

  # The gradient for the weights corresponding to the actual class is given by the sum over all other classes
  margins_boolean[list(range(num_train)), y] = -(np.sum(margins_boolean,axis=1) - 1)

  # dW = D * C, margins_boolean = N * C. Thus multiply by X^T to get dW = D * C
  dW = X.T.dot(margins_boolean)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW
