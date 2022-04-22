import os
from io import BytesIO
from typing import Dict

from jina import Executor, requests, DocumentArray, Document


def reload(mod_name: str):
    if f'{mod_name}_RELOADED' not in os.environ:
        print(f'import {mod_name}')
        import sys
        from jina.importer import PathImporter
        del sys.modules[mod_name]
        os.environ['LAZY_LOAD'] = '1'
        PathImporter.add_modules(f'{mod_name}.py')
        os.unsetenv('LAZY_LOAD')
        os.environ[f'{mod_name}_RELOADED'] = '1'
        print(f'import {mod_name} is done!')


class DalleMiniGenerator(Executor):

    @requests(on='/')
    def generate(self, docs: DocumentArray, parameters: Dict, **kwargs):
        reload('dm_helper')
        import dm_helper
        num_images = int(parameters.get('num_images', 1))
        with_caption = bool(parameters.get('caption', True))
        for d in docs:
            print(f'Created {num_images} images from text prompt [{d.text}]')
            generated_imgs = dm_helper.generate_images(d.text, num_images, caption=with_caption)

            for img in generated_imgs:
                buffered = BytesIO()
                img.save(buffered, format='JPEG')
                _d = Document(blob=buffered.getvalue(), mime_type='image/jpg').convert_blob_to_datauri()
                _d.blob = None
                d.chunks.append(_d)

            print(f'{d.text} done!')
