"""Another sample module for testing."""

from .sample_module import SampleClass


class AnotherClass:
    """Another test class."""
    
    def __init__(self):
        self.sample = SampleClass("another")
    
    def do_something(self):
        """Do something with the sample class."""
        return self.sample.public_method()