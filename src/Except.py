class NotFittedError(Exception):
    """Exception raised when solve() is called before fit()."""
    def __init__(self, message="This solver instance is not fitted yet. Call 'fit' first."):
        self.message = message
        super().__init__(self.message)

class GraphError(Exception):
    """Base class for graph exceptions."""
    pass

class NodeNotFoundError(GraphError):
    """Raised when a node index does not exist."""
    pass

class InvalidEdgeError(GraphError):
    """Raised when an edge refers to a non-existent node."""
    pass
