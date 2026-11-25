"""Local product checks using generated images and untrained CPU weights only."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from cs400.config import Candidate, SCHEMA_VERSION
from cs400.data import load_manifest, search_datasets, transform
from cs400.folder_data import prepare_folder, FolderDataset
from cs400.dashboard import make_server, Workspace
from cs400.inference import predict_image
from cs400.model import ResNet
from cs400.evaluation import evaluate_clean, evaluate_run
from cs400.storage import write_json


class FolderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(10)
        for split in ('train','validation','test'):
            for label in ('cat','dog'):
                directory = self.root / split / label
                directory.mkdir(parents=True)
                Image.fromarray(rng.integers(0,256,(40,48,3),dtype=np.uint8)).save(directory/'image.png')
        self.manifest_path = self.root/'manifest.json'

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        return prepare_folder(self.root,self.manifest_path,True)

    def test_loaders_do_not_read_test_and_preserve_class_mapping(self):
        manifest=self.prepare()
        self.assertEqual(load_manifest(self.manifest_path),manifest)
        for p in (self.root/'test').glob('*/*.png'):
            p.unlink()  # search must still work without any test images
        training,validation=search_datasets(self.root,manifest,'basic')
        self.assertEqual(training[0][0].shape,(3,32,32))
        self.assertEqual([validation[i][1] for i in range(2)],[0,1])
        with self.assertRaises(FileNotFoundError):FolderDataset(self.root,manifest,'test',transform(manifest))

    def test_cross_split_decoded_duplicate_rejected(self):
        with Image.open(self.root/'train/cat/image.png') as image:
            image.save(self.root/'test/cat/image.png',compress_level=1)
        with self.assertRaisesRegex(ValueError,'Identical decoded'):self.prepare()

    def test_modified_training_image_rejected(self):
        manifest=self.prepare()
        Image.new('RGB',(40,48)).save(self.root/'train/cat/image.png')
        with self.assertRaisesRegex(ValueError,'changed'):search_datasets(self.root,manifest,'basic')

    def test_classes_and_augmentation_acknowledgement(self):
        with self.assertRaises(ValueError):prepare_folder(self.root,self.manifest_path)
        (self.root/'test/dog').rename(self.root/'test/bird')
        with self.assertRaises(ValueError):self.prepare()

    def test_custom_clean_evaluation_and_cifar_corruption_rejection(self):
        manifest=self.prepare()
        fake_model=ResNet(Candidate(width=16,depth='1-1-1-1'),2).eval()
        meta={'protocol':{'batch_size':2},'candidate':asdict(Candidate())}
        checkpoint={'identity':'fixture','epoch':60}
        run=self.root/'run';run.mkdir();(run/'checkpoint.pt').write_bytes(b'fixture')
        with patch('cs400.evaluation.completed_model',return_value=(fake_model,meta,manifest,checkpoint)):
            result=evaluate_clean(run,self.manifest_path,self.root,self.root/'evaluation.json','cpu')
        self.assertEqual(result['samples'],2)
        self.assertEqual(result['conditions'],[])
        self.assertNotIn('corruption_accuracy',result['metrics'])
        with self.assertRaisesRegex(ValueError,'only valid for CIFAR'):
            evaluate_run(run,self.manifest_path,self.root,self.root,self.root/'bad.json','cpu')

    def test_portable_export_predictions_match_model(self):
        candidate=Candidate(width=16,depth='1-1-1-1')
        model=ResNet(candidate,2).eval()
        path=self.root/'export.pt'
        torch.save(dict(schema=SCHEMA_VERSION,kind='cs400-model-export',model=model.state_dict(),
                        candidate=asdict(candidate),classes=['cat','dog'],
                        preprocessing={'image_size':32,'mean':[.5]*3,'std':[.5]*3}),path)
        result=predict_image(path,self.root/'test/cat/image.png')
        self.assertEqual(len(result['top_predictions']),2)
        self.assertAlmostEqual(sum(p['probability'] for p in result['top_predictions']),1,places=6)
        manifest=self.prepare()
        tensor=FolderDataset(self.root,manifest,'test',transform(manifest))[0][0].unsqueeze(0)
        with torch.inference_mode():expected=model(tensor).softmax(1)[0]
        for row in result['top_predictions']:
            self.assertAlmostEqual(row['probability'],expected[['cat','dog'].index(row['label'])].item(),places=6)


class DashboardTests(unittest.TestCase):
    def test_http_read_only_routes_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            server=make_server(temp,0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            address=f'http://127.0.0.1:{server.server_address[1]}'
            try:
                with urlopen(address+'/api/runs') as response:self.assertEqual(json.load(response),[])
                with urlopen(address) as response:self.assertIn(b'ResNet search workspace',response.read())
                with self.assertRaises(HTTPError) as blocked:urlopen(address+'/download?path=../../README.md')
                self.assertEqual(blocked.exception.code,400)
                request=Request(address+'/api/export',data=b'{}',headers={'Content-Type':'application/json'},method='POST')
                with self.assertRaises(HTTPError) as blocked:urlopen(request)
                self.assertEqual(blocked.exception.code,403)
                request=Request(address,headers={'Host':'attacker.example'})
                with self.assertRaises(HTTPError) as blocked:urlopen(request)
                self.assertEqual(blocked.exception.code,403)
                with self.assertRaises(ValueError):Workspace(temp).run_directory('../outside')
            finally:
                server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
