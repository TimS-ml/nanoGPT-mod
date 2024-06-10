def add_to_class(Class):
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


class FunctionBase(object):
    def __call__(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.fn(x)

    def fn(self, x):
        raise NotImplementedError()

    def grad(self, x):
        raise NotImplementedError()
