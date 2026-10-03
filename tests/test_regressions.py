import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
ROOT = Path(__file__).resolve().parents[1]

def definitions(filename, names, namespace):
    path = ROOT / filename
    if path.suffix == '.ipynb':
        nb = json.loads(path.read_text())
        text = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        text = '\n'.join(line if not line.startswith(('!', '%', 'pip install')) else '# '+line for line in text.splitlines())
    else:
        text = path.read_text()
    tree = ast.parse(text)
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random, heapq, time
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def encode(self, text): return SimpleNamespace(ids=[int(t) for t in text.split()])
    def get_vocab_size(self): return 5
    def decode(self, ids): return ' '.join(map(str,ids))
    def token_to_id(self, text): return 0

NS = definitions('Prof_Faith_Text_Generation_with_RNNs.ipynb',
    {'RNNModel','SubwordTextDataset','evaluate','train','generate_text_temperature','generate_text_beam_search'},
    dict(torch=torch,nn=nn,F=F,Dataset=Dataset,math=math,random=random,heapq=heapq,time=time,
         device=torch.device('cpu'),config=SimpleNamespace(seq_len=3,log_interval=50)))

class RecurrentRegressionTests(unittest.TestCase):
    def test_all_recurrent_architectures_and_small_final_batch(self):
        for kind in ['rnn','lstm','gru']:
            model=NS['RNNModel'](5,4,6,1,dropout=0.,model_type=kind)
            ds=NS['SubwordTextDataset']('0 1 2 3 4 0 1 2',3,Tokenizer())
            loader=DataLoader(ds,batch_size=4)
            loss,ppl=NS['evaluate'](model,loader,nn.CrossEntropyLoss(),5)
            self.assertTrue(math.isfinite(loss) and math.isfinite(ppl))
            optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            NS['train'](model,loader,1,nn.CrossEntropyLoss(),optimizer,1.,5)
    def test_bidirectional_rejected_for_causal_generation(self):
        with self.assertRaises(ValueError): NS['RNNModel'](5,4,6,1,bidirectional=True)
    def test_empty_evaluation_rejected(self):
        model=NS['RNNModel'](5,4,6,1)
        with self.assertRaises(ValueError): NS['evaluate'](model,[],nn.CrossEntropyLoss(),5)
    def test_lazy_dataset_preserves_shifted_targets(self):
        ds=NS['SubwordTextDataset']('0 1 2 3 4',3,Tokenizer())
        x,y=ds[1]
        self.assertEqual(x.tolist(),[1,2,3]);self.assertEqual(y.tolist(),[2,3,4])
        self.assertFalse(hasattr(ds,'sequences'))
    def test_sampling_validates_temperature(self):
        model=NS['RNNModel'](5,4,6,1)
        for temp in [0.,-1.,float('nan')]:
            with self.assertRaises(ValueError): NS['generate_text_temperature'](model,'1 2',1,temp,Tokenizer(),'cpu')
    def test_decoders_consume_prompt_once(self):
        class RecordingModel:
            def __init__(self): self.inputs=[]
            def eval(self): pass
            def init_hidden(self,n): return torch.zeros(1,1,1)
            def __call__(self,x,h):
                self.inputs.extend(x.flatten().tolist())
                logits=torch.zeros(1,x.shape[1],5);logits[:,:,-1]=10000
                return logits,h
        for function,args in [(NS['generate_text_temperature'],(1.,)),(NS['generate_text_beam_search'],(1,))]:
            model=RecordingModel(); text=function(model,'1 2 3',1,*args,Tokenizer(),'cpu')
            self.assertEqual(model.inputs,[1,2,3]);self.assertEqual(text,'1 2 3 4')
