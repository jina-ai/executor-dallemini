# DalleMini

Generating image from text prompt using DALLE-mini model.

- Incoming Document must have `.text` filled.
- Returned image Documents are nested under each `.chunks`'s `.uri` as base64 DataURI jpg format.
- One can use `parameters={'num_images': 1}` to set the number of images returned, 1 image for 1 Document takes ~30 seconds on CPU.

## Caveat

Memory demanding!

First request would take ~2min, the proceeding request would take 30 second/each.

The reason is that Jina <=3.3.1's `@request` function is living in another thread than other `def` or `__init__`, making all JAX JIT functions unusable. To solve that a very hacky solution is implemented to "reload" the module on the first request, to ensure that JAX JIT functions are always living in the context of request caller, not in other thread.

This is considered as a design flaw in Jina <=3.3.1 and shall be fixed soon.

