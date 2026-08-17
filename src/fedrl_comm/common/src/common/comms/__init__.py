"""
Agent <-> server communication channel abstraction.

Every algorithm (fednpg_admm, baselines) should send/receive through this
module rather than passing tensors directly, so that:
  1. communication byte-counts can be measured consistently across
     algorithms (needed for the O(d) vs O(d^2) comparison in the report)
  2. the `quantization` package can wrap this channel and quantize whatever
     passes through it, without either algorithm package needing to know
     quantization exists

TODO: implement a Channel class with .send(tensor) / .receive() and a
      byte-accounting hook that `quantization` and `results` can both read.
"""
