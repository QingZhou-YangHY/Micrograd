class Value:
    
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        """
        当你写 2 + v 时，Python 先尝试 int.__add__(2, v)，通常不认识你的 Value，
        然后会回退调用 v.__radd__(2)
        """
        return self + other

    def __rmul__(self, other):
        """
        当你写 2 * v 时，Python 先尝试 int.__mul__(2, v)，通常不认识你的 Value，
        然后会回退调用 v.__rmul__(2)
        """
        return self * other

    # 实现更通用"除法"的思路: a / b = a * (1/b) = a * (b**-1)
    # 这样我们只需要实现幂运算就可以了,当K = -1时就是除法
    def __truediv__(self, other):
        return self * other**-1    

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        """
        自动化求导算法的核心是拓扑排序（topological sort）。
        我们需要按照计算图的反向顺序来调用每个节点的 _backward 方法，
        以确保在计算梯度时，所有依赖的节点都已经计算好了梯度。
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
