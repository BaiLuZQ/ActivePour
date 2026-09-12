"""Synthetic CPU smoke tests, including exact optimizer-boundary resume."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from activepour.io import save
from activepour.train_readout import main

class TrainingTests(unittest.TestCase):
    def test_three_arms_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);bank=root/'bank';bank.mkdir();torch.manual_seed(41)
            for split in ['train','validation']:
                save(bank/(split+'.pt'),dict(split=split,ids=[split+'0',split+'1'],episode_ids=[split+'ep0',split+'ep1'],
                     probe=torch.randn(2,5,256,256),action=torch.tensor([[65.,1.,.6],[75.,1.2,.4]]),
                     eta=torch.tensor([.2,.7]),x=torch.randn(2,1664),parent_sha256='synthetic-parent'))
            def run(arm,name,*extra):
                args=['train_readout','--bank',str(bank),'--arm',arm,'--out',str(root/name),'--steps','3','--device','cpu',*extra]
                with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):main()
            run('direct','full');run('direct','resumed','--stop-after','1');run('direct','resumed','--resume')
            a=torch.load(root/'full/latest.pt',weights_only=True,map_location='cpu')
            b=torch.load(root/'resumed/latest.pt',weights_only=True,map_location='cpu')
            for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
            for arm in ['latent','zero']:
                run(arm,arm)
                state=torch.load(root/arm/'latest.pt',weights_only=True,map_location='cpu')
                self.assertEqual(state['step'],3);self.assertEqual(state['normalization']['fit_split'],'train')

if __name__=='__main__':unittest.main()
