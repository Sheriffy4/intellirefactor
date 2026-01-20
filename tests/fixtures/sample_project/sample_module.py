"""
Sample module for testing expert analysis.

This is a simple test module to verify the analysis pipeline works correctly.
"""

class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, name: str):
        """Initialize the sample class."""
        self.name = name
    
    def public_method(self) -> str:
        """A public method that does something."""
        return f"Hello, {self.name}!"
    
    def _private_method(self) -> None:
        """A private helper method."""
        pass


def sample_function(x: int, y: int) -> int:
    """A sample function for testing."""
    return x + y


if __name__ == "__main__":
    # Entry point for testing
    obj = SampleClass("test")
    result = obj.public_method()
    print(result)