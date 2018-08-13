import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class AffineEndCNN(object):
    """
    [conv-sbatchnorm-relu-pool]xN - [affine-batchnorm]xM - affine - softmax
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_sizes=[4], hidden_dims=[100], num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=True, dtype=np.float32):
        self.params = {}
        self.use_batchnorm = use_batchnorm
        self.reg = reg
        self.dtype = dtype
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.layer_dims = hidden_dims + [num_classes]
        C, H, W = input_dim
        out_size = [H, W]
        prev_num_filter = C
        for idx, num_filter in enumerate(num_filters):
            dictID = idx + 1
            self.params['W{}'.format(dictID)] = np.random.randn(
                num_filter, prev_num_filter, filter_sizes[idx], filter_sizes[idx]) * weight_scale
            prev_num_filter = num_filter
            self.params['b{}'.format(dictID)] = np.zeros(num_filter)
            if use_batchnorm:
                self.params['gamma{}'.format(dictID)] = np.ones((num_filter))
                self.params['beta{}'.format(dictID)] = np.zeros((num_filter))
            out_size = [x//2 for x in out_size]

        prev_dim = np.prod(out_size) * num_filters[-1]
        for idx, hidden_dim in enumerate(self.layer_dims):
            dictID = idx + len(num_filters) + 1
            self.params['W{}'.format(dictID)] = np.random.randn(
                prev_dim, hidden_dim) * weight_scale
            self.params['b{}'.format(dictID)] = np.zeros(hidden_dim)
            prev_dim = hidden_dim
            if use_batchnorm and idx != len(self.layer_dims) - 1:
                self.params['gamma{}'.format(dictID)] = np.ones((hidden_dim))
                self.params['beta{}'.format(dictID)] = np.zeros((hidden_dim))

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in xrange(len(num_filters) + len(hidden_dims))]

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode
        scores = None
        caches = []
        layer_out = X
        N = len(self.num_filters)
        M = len(self.layer_dims)
        for idx in xrange(N):
            conv_param = {'stride': 1, 'pad': (self.filter_sizes[idx] - 1) / 2}
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
            dictID = idx + 1
            W, b = self.params['W{}'.format(
                dictID)], self.params['b{}'.format(dictID)]
            if self.use_batchnorm:
                gamma, beta = self.params['gamma{}'.format(
                    dictID)], self.params['beta{}'.format(dictID)]
                layer_out, cache = conv_norm_relu_pool_forward(
                    layer_out, W, b, gamma, beta, conv_param, self.bn_params[idx], pool_param)
            else:
                layer_out, cache = conv_relu_pool_forward(
                    layer_out, W, b, conv_param, pool_param)
            caches.append(cache)

        for idx in xrange(M):
            dictID = idx + N + 1
            W, b = self.params['W{}'.format(
                dictID)], self.params['b{}'.format(dictID)]
            if idx == M - 1:
                layer_out, cache = affine_forward(layer_out, W, b)
            elif self.use_batchnorm:
                gamma, beta = self.params['gamma{}'.format(
                    dictID)], self.params['beta{}'.format(dictID)]
                layer_out, cache = affine_norm_relu_forward(
                    layer_out, W, b, gamma, beta, self.bn_params[N + idx])
            else:
                layer_out, cache = affine_relu_forward(layer_out, W, b)
            caches.append(cache)
        scores = layer_out

        if mode == 'test':
            return scores

        grads = {}
        loss, dout = softmax_loss(scores, y)
        for idx in xrange(M):
            dictID = N + M - idx
            W, b = self.params['W{}'.format(
                dictID)], self.params['b{}'.format(dictID)]
            loss += 0.5 * self.reg * np.sum(W*W)
            if idx == 0:
                dout, dw, db = affine_backward(dout, caches[-idx - 1])
            elif self.use_batchnorm:
                dout, dw, db, dgamma, dbeta = affine_norm_relu_backward(
                    dout, caches[-idx-1])
                grads['gamma{}'.format(dictID)] = dgamma
                grads['beta{}'.format(dictID)] = dbeta
            else:
                dout, dw, db = affine_relu_backward(dout, caches[-idx-1])
            grads['W{}'.format(dictID)] = dw + self.reg * W
            grads['b{}'.format(dictID)] = db
        for idx in xrange(N - 1, -1, -1):
            dictID = idx + 1
            W, b = self.params['W{}'.format(
                dictID)], self.params['b{}'.format(dictID)]
            loss += 0.5 * self.reg * np.sum(W*W)
            if self.use_batchnorm:
                dout, dw, db, dgamma, dbeta = conv_norm_relu_pool_backward(
                    dout, caches[idx])
                grads['gamma{}'.format(dictID)] = dgamma
                grads['beta{}'.format(dictID)] = dbeta
            else:
                dout, dw, db = conv_relu_pool_backward(dout, caches[idx])
            grads['W{}'.format(dictID)] = dw + self.reg*W
            grads['b{}'.format(dictID)] = db
        return loss, grads


def conv_norm_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    n, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(n)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache


def conv_norm_relu_pool_backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    dn = relu_backward(ds, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dn, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
    a, fc_cache = affine_forward(x, w, b)
    z, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(z)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


def affine_norm_relu_backward(dout, cache):
    fc_cache, bn_cache, relu_cache = cache
    dz = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dz, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta
