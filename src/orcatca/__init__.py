"""Acoustic domain-generalization experiments."""
import os

# Import the NPU backend only after runtime checks and logging are configured.
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
__version__ = '2.3.0'
