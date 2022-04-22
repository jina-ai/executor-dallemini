# DalleMini

Generating image from text prompt using DALLE-mini model.

- Incoming Document must have `.text` filled.
- Returned image Documents are nested under each `.chunks`'s `.uri` as base64 DataURI jpg format.
- One can use `parameters={'num_images': 1}` to set the number of images returned, 1 image for 1 Document takes ~30 seconds on CPU.
- One can turn off captioning by `parameters={'caption': False}`.

## Caveat

Memory demanding!

Each request would take 30 second/each.

