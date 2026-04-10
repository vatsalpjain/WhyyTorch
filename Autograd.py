"""A tiny NumPy-based autograd and MLP library.

This module provides a scalar/tensor wrapper with reverse-mode automatic
differentiation, a linear layer, a simple multilayer perceptron, and a mean
square error loss helper.
"""

import numpy as np

class WhyyTorch:
    """A NumPy-backed tensor that tracks operations for reverse-mode autograd."""
    
    def __init__(self, data, requires_grad=True, _op="", children=(), label=None):
        """Create a tensor wrapper.

        Args:
            data: Array-like input data.
            requires_grad: Whether to allocate and track gradients.
            _op: Internal operation name used for graph inspection.
            children: Parent tensors in the computation graph.
            label: Optional human-readable label for debugging.
        """
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float32,) if requires_grad else None
        self.op = _op
        self.children = tuple(children)
        self.label = label
        self._backward = lambda: None
    
        if self.requires_grad and self.data.shape != self.grad.shape:
            raise ValueError(
                f"Data and grad must have the same shape, got {self.data.shape} and {self.grad.shape}"
            )
        
    def __repr__(self):
        return f"Label: {self.label} Value: {self.data} Grad: {self.grad}"
    
    @staticmethod
    def matrix_multiply(a, b):
        """Multiply two tensors and register the backward pass."""
        assert a.data.ndim >= 1 and b.data.ndim >= 1, "Matmul requires tensors with ndim >= 1" 
        assert a.data.shape[-1] == b.data.shape[-2] if b.data.ndim >= 2 else a.data.shape[-1] == b.data.shape[0], "Incompatible shapes for matrix multiplication"
        x = WhyyTorch(
            a.data @ b.data,
            requires_grad=a.requires_grad or b.requires_grad,
            _op="@",
            children=(a, b),
        )
        
        def backward():
            a_data = a.data
            b_data = b.data
            grad_out = x.grad

            a_was_1d = (a_data.ndim == 1)
            b_was_1d = (b_data.ndim == 1)

            a_work = a_data[np.newaxis, :] if a_was_1d else a_data
            b_work = b_data[:, np.newaxis] if b_was_1d else b_data

            if a_was_1d and b_was_1d:
                grad_work = np.array(grad_out, dtype=np.float32).reshape(1, 1)
            elif a_was_1d:
                grad_work = np.expand_dims(grad_out, axis=-2)
            elif b_was_1d:
                grad_work = np.expand_dims(grad_out, axis=-1)
            else:
                grad_work = grad_out

            b_t = np.swapaxes(b_work, -1, -2)
            a_t = np.swapaxes(a_work, -1, -2)

            if a.requires_grad:
                grad_a = grad_work @ b_t
                if a_was_1d:
                    grad_a = np.squeeze(grad_a, axis=-2)
                a._accumulate_grad(grad_a)

            if b.requires_grad:
                grad_b = a_t @ grad_work
                if b_was_1d:
                    grad_b = np.squeeze(grad_b, axis=-1)
                b._accumulate_grad(grad_b)

        x._backward = backward
        return x
    
    @staticmethod
    def _sum_to_shape(grad, shape):
        """Reduce a broadcast gradient back to the requested shape."""
        g = np.array(grad, dtype=np.float32)
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for axis, size in enumerate(shape):
            if size == 1 and g.shape[axis] != 1:
                g = g.sum(axis=axis, keepdims=True)
        return g
    
    def _accumulate_grad(self, grad):
        """Accumulate a gradient into this tensor, handling broadcasting."""
        if not self.requires_grad:
            return
        self.grad += WhyyTorch._sum_to_shape(grad, self.data.shape)

    def zero_grad(self):
        """Reset the gradient buffer to zeros."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def _coerce(self, other):
        """Convert a Python scalar or array-like value into a tensor."""
        return other if isinstance(other, WhyyTorch) else WhyyTorch(other, requires_grad=False)

    def sum(self, axis=None, keepdims=False):
        """Sum the tensor, preserving the computation graph."""
        out = WhyyTorch(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op="sum",
            children=(self,),
        )
        def backward():
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.data.shape)
            self._accumulate_grad(grad)
        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Compute the mean of the tensor, preserving the computation graph."""
        out = WhyyTorch(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _op="mean",
            children=(self,),
        )
        def backward():
            grad = out.grad
            if axis is None:
                count = self.data.size
                grad = np.broadcast_to(grad / count, self.data.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in axes)
                count = 1
                for ax in axes:
                    count *= self.data.shape[ax]
                if not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad / count, self.data.shape)
            self._accumulate_grad(grad)
        out._backward = backward
        return out

    def relu(self):
        """Apply ReLU elementwise."""
        out = WhyyTorch(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _op="relu",
            children=(self,),
        )
        def backward():
            self._accumulate_grad((self.data > 0).astype(np.float32) * out.grad)
        out._backward = backward
        return out

    def tanh(self):
        """Apply tanh elementwise."""
        t = np.tanh(self.data)
        out = WhyyTorch(
            t,
            requires_grad=self.requires_grad,
            _op="tanh",
            children=(self,),
        )
        def backward():
            self._accumulate_grad((1.0 - t ** 2) * out.grad)
        out._backward = backward
        return out

    def __add__(self, other):
        """Add two tensors."""
        other = self._coerce(other)
        x = WhyyTorch(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="+",
            children=(self, other),
        )
        def backward():
            self._accumulate_grad(x.grad)
            other._accumulate_grad(x.grad)
        x._backward = backward
        return x

    def __sub__(self, other):
        """Subtract one tensor from another."""
        other = self._coerce(other)
        x = WhyyTorch(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="-",
            children=(self, other),
        )
        def backward():
            self._accumulate_grad(x.grad)
            other._accumulate_grad(-x.grad)
        x._backward = backward
        return x

    def __mul__(self, other):
        """Multiply two tensors elementwise."""
        other = self._coerce(other)
        x = WhyyTorch(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="*",
            children=(self, other),
        )
        def backward():
            self._accumulate_grad(other.data * x.grad)
            other._accumulate_grad(self.data * x.grad)
        x._backward = backward
        return x

    def __truediv__(self, other):
        """Divide one tensor by another elementwise."""
        other = self._coerce(other)
        x = WhyyTorch(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _op="/",
            children=(self, other),
        )
        def backward():
            self._accumulate_grad((1 / other.data) * x.grad)
            other._accumulate_grad((-self.data / (other.data ** 2)) * x.grad)
        x._backward = backward
        return x
    
    def __pow__(self, p):
        """Raise the tensor to a scalar power."""
        assert isinstance(p, (int, float))
        x = WhyyTorch(
            self.data ** p,
            requires_grad=self.requires_grad,
            _op=f"**{p}",
            children=(self,),
        )
        def backward():
            self._accumulate_grad((p * (self.data ** (p - 1))) * x.grad)
        x._backward = backward
        return x

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __matmul__(self, other):
        return WhyyTorch.matrix_multiply(self, self._coerce(other))

    def __rmatmul__(self, other):
        return WhyyTorch.matrix_multiply(self._coerce(other), self)
    
    def backward(self, grad=None):
        """Run reverse-mode autodiff from this tensor back through the graph."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Non-scalar outputs need an explicit grad argument.")
            grad = np.ones_like(self.data, dtype=np.float32)
        else:
            grad = np.array(grad, dtype=np.float32)
            if grad.shape != self.data.shape:
                raise ValueError(f"Backward grad shape {grad.shape} must match output shape {self.data.shape}")

        self._accumulate_grad(grad)
        for v in reversed(topo):
            v._backward()
    
class Linear:
    """A fully connected layer using WhyyTorch parameters."""

    def __init__(self, in_features, out_features):
        """Create a linear layer with He-initialized weights."""
        self.w = WhyyTorch(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self.b = WhyyTorch(np.zeros((out_features,), dtype=np.float32))
    
    def __call__(self, x):
        """Apply the affine transform xW + b."""
        return x @ self.w + self.b
    
    def parameters(self):
        """Return the trainable parameters for this layer."""
        return [self.w, self.b]


class MLP:
    """A simple feed-forward multilayer perceptron with ReLU activations.
    Args: 
    in_features: The number of input features.
    layer_sizes: A list of integers specifying the width of each hidden layer and output layer.
    output: The output of the final layer, with no activation applied.
    """

    def __init__(self, in_features, layer_sizes):
        """Create an MLP from an input size and a list of layer widths."""
        sizes = [in_features] + list(layer_sizes)
        self.layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]
    
    def __call__(self, x):
        """Run a forward pass through the network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x
    
    def parameters(self):
        """Return all trainable parameters in the network."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for every parameter in the network."""
        for param in self.parameters():
            param.zero_grad()
    
    def step(self, lr):
        """Apply an SGD update to every parameter."""
        for param in self.parameters():
            param.data -= lr * param.grad
        
def mse_loss(pred, target,epoch=0):
    """Return mean squared error between predictions and targets."""
    if epoch % 100 ==0:
        print(f"Loss: {((pred - target) ** 2).mean().data}")
    return ((pred - target) ** 2).mean()

__all__ = ["WhyyTorch", "Linear", "MLP", "mse_loss"]