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
    N = np.shape(X)[0]
    D = np.shape(W)[0]
    C = np.shape(W)[1]

    # ====== These explicit loops are too slow ======
    # for i in range(N):
    #     z = W.T.dot(X[i, :])
    #     numerator = np.exp(z)
    #     denominator = np.sum(numerator)
    #     y_ = numerator / denominator
    #     loss += -np.log(numerator[y[i]] / denominator) / N
    #     for d in range(D):
    #         for c in range(C):
    #             dW[d, c] += X[i, d] * (y_[c] - (y[i] == c)) / N
    for i in range(N):
        z = X[i, :].dot(W)
        z -= np.max(z)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        y_ = numerator / denominator
        loss += np.sum(-np.log(y_[y[i]]) / N)
        for c in range(C):
            dW[:, c] += X[i, :] * (y_[c] - (y[i] == c)) / N
    loss += 0.5 * reg * np.trace(W.T.dot(W))
    dW += reg * W
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
    N = np.shape(X)[0]
    z = X.dot(W)
    z -= np.max(z, axis=1)[:, np.newaxis]
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1)
    y_ = numerator / denominator[:, np.newaxis]
    loss = np.sum(-np.log(y_[np.arange(N), y]) / N)
    loss += 0.5 * reg * np.trace(W.T.dot(W))
    y_[np.arange(N), y] -= 1
    dW = X.T.dot(y_) / N + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
