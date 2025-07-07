class Node:
    def __init__(self, grad_fn, next_functions, name=""):
        """
        A class representing a gradient function node in the computational graph.
        Gradient function nodes encapsulate the gradient computation and propagation
        for a specific operation in the graph.

        Args:
            grad_fn: The gradient function.
            next_functions: A tuple of next gradient function nodes.
            name: The name of the gradient function node (optional).
        """
        self.grad_fn = grad_fn
        self.next_functions = next_functions
        self.name = name

    def __call__(self, grad):
        """
        Call the gradient function with the given gradient.

        Args:
            grad: The gradient to be passed to the gradient function.

        Returns:
            The result of the gradient function.
        """
        if self.grad_fn:
            return self.grad_fn(grad)
        else:
            raise RuntimeError("Trying to backward through the graph a second time.")
    
    def __repr__(self):
        """
        Return a string representation of the gradient function node.

        Returns:
            A string representation of the gradient function node.
        """
        return f"Node={self.name}"