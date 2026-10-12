"""
quantization: quantize the agent->server transmissions of FedNPG-ADMM and
measure performance against bit width. Not part of the original paper.

quantizers.py holds the schemes (stochastic uniform, reduced-precision
float); sweep.py is the fedrl-quantize-sweep CLI.
"""

__version__ = "0.1.0"
