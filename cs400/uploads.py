"""Bounded ZIP ingestion into an immutable local image-classification dataset."""
from collections import defaultdict
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import tempfile
import uuid
import warnings
import zipfile
from PIL import Image
from .folder_data import prepare_folder, image_record, SPLITS, EXTENSIONS
from .storage import write_json

MAX_UPLOAD = 512 * 1024**2
MAX_EXPANDED = 2 * 1024**3
MAX_FILES = 100_000
MAX_IMAGE = 20 * 1024**2


def safe_parts(name):
    parts = PurePosixPath(name).parts
    if not parts or name.startswith('/') or '\\' in name:
        raise ValueError('ZIP entries must use relative folder/file paths')
    for part in parts:
        if part in ('.', '..') or not re.fullmatch(r'[\w .()-]+', part, re.UNICODE) or part.endswith((' ', '.')):
            raise ValueError(f'Unsupported or unsafe ZIP path: {name}')
        if part.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
            raise ValueError('Reserved filename in ZIP')
    return parts


def import_zip(root, archive_path, *, augmentation_confirmed=False):
    if augmentation_confirmed is not True:
        raise ValueError('Confirm that resizing, crop, flip and erasing preserve the class labels')
    root = Path(root).resolve()
    destination = root/'data'/'uploads';destination.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='import-', dir=destination) as temp:
        stage = Path(temp);raw=stage/'raw';images=stage/'images'
        with zipfile.ZipFile(archive_path) as archive:
            entries=[];seen=set();expanded=0
            for item in archive.infolist():
                parts=safe_parts(item.filename.rstrip('/'))
                mode=item.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)):
                    raise ValueError('Links and special files are not supported in ZIP uploads')
                if item.is_dir():continue
                if item.flag_bits & 1:raise ValueError('Encrypted ZIP files are not supported')
                key='/'.join(parts).casefold()
                if key in seen:raise ValueError('Duplicate or case-conflicting ZIP paths')
                seen.add(key)
                if PurePosixPath(parts[-1]).suffix.lower() not in EXTENSIONS:
                    raise ValueError('Only PNG/JPEG image files are accepted; remove other files from the ZIP')
                if not 0 < item.file_size <= MAX_IMAGE:raise ValueError('Each image must be between 1 byte and 20 MiB')
                expanded+=item.file_size;entries.append((item,parts))
                if len(entries)>MAX_FILES or expanded>MAX_EXPANDED:raise ValueError('Dataset exceeds the 100,000-file / 2 GiB expanded limit')
            if not entries:raise ValueError('ZIP contains no supported images')
            # Optional single outer dataset directory.
            paths=[p for _,p in entries]
            if len({p[0] for p in paths})==1 and all(len(p)>=3 for p in paths):
                paths=[p[1:] for p in paths]
            explicit=all(len(p)==3 and p[0] in SPLITS for p in paths)
            automatic=all(len(p)==2 for p in paths)
            if not (explicit or automatic):
                raise ValueError('Use class/image.png folders, or train/class/image.png, validation/class/image.png and test/class/image.png')
            classes=sorted({p[-2] for p in paths})
            if not 2<=len(classes)<=100 or len({c.casefold() for c in classes})!=len(classes):
                raise ValueError('Use 2 to 100 distinct class folders (case-insensitive names must differ)')
            groups=defaultdict(list);pixels_seen=set()
            for index,((item,_),parts) in enumerate(zip(entries,paths)):
                path=raw/(str(index)+PurePosixPath(parts[-1]).suffix.lower());path.parent.mkdir(exist_ok=True)
                with archive.open(item) as src,path.open('wb') as dst:
                    remaining=item.file_size
                    while chunk:=src.read(min(1024**2,remaining+1)):
                        remaining-=len(chunk)
                        if remaining<0:raise ValueError('ZIP size does not match entry metadata')
                        dst.write(chunk)
                    if remaining:raise ValueError('Truncated ZIP image')
                with warnings.catch_warnings():
                    warnings.simplefilter('error',Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        if image.width*image.height>16_000_000:raise ValueError('Images must be at most 16 megapixels')
                    _,pixel_hash=image_record(path)
                if automatic and pixel_hash in pixels_seen:
                    raise ValueError('Automatic splitting requires distinct decoded images; remove duplicates or supply verified splits')
                pixels_seen.add(pixel_hash)
                groups[(parts[0] if explicit else None,parts[-2])].append(path)
            for (split,label),files in groups.items():
                if automatic:
                    if len(files)<10:raise ValueError('Automatic splitting needs at least 10 images per class')
                    random.Random(42).shuffle(files)
                    held=max(1,int(len(files)*.15))
                    buckets={'validation':files[:held],'test':files[held:2*held],'train':files[2*held:]}
                else:buckets={split:files}
                for name,items in buckets.items():
                    folder=images/name/label;folder.mkdir(parents=True,exist_ok=True)
                    for file in items:shutil.move(str(file),str(folder/file.name))
        manifest=prepare_folder(images,stage/'manifest.json',True)
        if len(manifest['train'])<4:raise ValueError('At least four training images are required')
        record=dict(id=uid,classes=manifest['classes'],counts={s:len(manifest[s]) for s in SPLITS},
                    split_policy='provided splits' if explicit else 'stratified approximately 70/15/15; seed 42',
                    manifest=f'data/uploads/{uid}/manifest.json',data_root=f'data/uploads/{uid}/images',
                    preprocessing='RGB; bilinear resize to 32 x 32; fixed mean/std 0.5',
                    corruption_evaluation='Not available for custom datasets without a domain benchmark')
        write_json(stage/'dataset.json',record)
        # Publish only fully validated datasets; raw extraction is not part of the dataset.
        target=destination/uid;target.mkdir()
        (stage/'images').rename(target/'images')
        (stage/'manifest.json').rename(target/'manifest.json')
        (stage/'dataset.json').rename(target/'dataset.json')
    return record
