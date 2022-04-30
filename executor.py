from io import BytesIO
from typing import Dict

from jina import Executor, requests, DocumentArray, Document

import dm_helper


class DalleMiniGenerator(Executor):

    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):

        num_images = int(parameters.get('num_images', 1))
        for d in docs:
            print(f'Created {num_images} images from text prompt [{d.text}]')
            generated_imgs = dm_helper.generate_images(d.text, num_images)

            for img in generated_imgs:
                buffered = BytesIO()
                img.save(buffered, format='PNG')
                _d = Document(blob=buffered.getvalue(), mime_type='image/png').convert_blob_to_datauri()
                _d.blob = None
                d.chunks.append(_d)

            print(f'{d.text} done!')
