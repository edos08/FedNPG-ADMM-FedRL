"""
Quantization schemes for tensors passed through common.comms.Channel.

TODO: implement e.g.
    def stochastic_quantize(tensor, num_bits) -> quantized_tensor
    def dequantize(quantized_tensor, meta) -> approx_tensor

Read the paper's own citation [9] (Q-GADMM: quantized group ADMM) before
inventing a scheme from scratch — it's solving close to the same problem.
"""
