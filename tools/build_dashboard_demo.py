"""Build a labelled synthetic upload example; no download or training."""
from pathlib import Path
import io
import random
import zipfile
from PIL import Image, ImageDraw


def build(root):
    output=Path(root)/'output/demo';output.mkdir(parents=True,exist_ok=True)
    target=output/'synthetic-shapes-demo.zip'
    rng=random.Random(42)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for label in ('circle','square'):
            for index in range(200):
                image=Image.new('RGB',(32,32),tuple(rng.randrange(15,65) for _ in range(3)))
                draw=ImageDraw.Draw(image)
                x,y=rng.randrange(6,10),rng.randrange(6,10);size=rng.randrange(15,20)
                color=tuple(rng.randrange(160,256) for _ in range(3))
                shape=draw.ellipse if label=='circle' else draw.rectangle
                shape((x,y,min(x+size,29),min(y+size,29)),fill=color)
                split='train' if index<140 else 'validation' if index<170 else 'test'
                stream=io.BytesIO();image.save(stream,format='PNG')
                archive.writestr(f'{split}/{label}/{index:03d}.png',stream.getvalue())
    (output/'README.md').write_text('''# Synthetic dashboard demonstration

`synthetic-shapes-demo.zip` contains 400 generated RGB images labelled circle or square. Each class has 140 training, 30 validation and 30 test images. It uses a fixed seed of 42 and contains no external dataset or personal information.

Upload the ZIP, confirm the transformations, select **Short workflow demonstration**, and press **Start local search** manually. Calibration checks whether the budget can cover startup and one cohort; increase the budget if requested. Use Pause and Resume as needed.

This example checks the upload-to-model workflow only. It is not a realistic benchmark, an additional academic experiment, or evidence of search superiority. The four historical CIFAR-10 experiments remain separate. This file contains no pretrained models.
''',encoding='utf-8')
    print(target)


if __name__=='__main__':build(Path.cwd())
