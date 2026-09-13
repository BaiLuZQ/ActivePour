"""CPU-only architecture and protocol tests; no model download required."""
import unittest
import torch
from activepour_model.conditioning import PhysicsConditioner,stage_grid
from activepour.readout import Head,Direct,pooled_future,normalize
from activepour.search import tests as search_tests

class CoreTests(unittest.TestCase):
    def test_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in Head().parameters()),221441)
        self.assertEqual(sum(p.numel() for p in Direct().parameters()),397441)

    def test_latent_quadrants(self):
        z=torch.zeros(1,16,64,64)
        for k,(y,x) in enumerate([(0,0),(0,32),(32,0),(32,32)]):z[:,:,y:y+32,x:x+32]=k+1
        expected=torch.arange(1.,5.)[:,None,None,None].expand(4,16,4,4)
        torch.testing.assert_close(pooled_future(z).reshape(4,16,4,4),expected)

    def test_pre1_does_not_read_hidden_frames(self):
        torch.set_num_threads(2);m=PhysicsConditioner(frame_mode='pre1').eval()
        p=torch.randn(1,5,256,256);q=p.clone();q[:,1:]=float('nan');a=torch.tensor([[65.,1.,.6]])
        with torch.no_grad():x=m.temporal_features(p,a);y=m.temporal_features(q,a)
        torch.testing.assert_close(x[0],y[0]);self.assertEqual(float(y[2].abs().max()),0)

    def test_condition_shapes_and_nominal_parameters(self):
        a=PhysicsConditioner(frame_mode='full5');b=PhysicsConditioner(frame_mode='pre1')
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        with torch.no_grad():h,_,_=a.eval().temporal_features(torch.zeros(1,5,256,256),torch.tensor([[65.,1.,.6]]))
        self.assertEqual(h.shape,(1,4,128,16,16));self.assertEqual(stage_grid(h).shape,(1,128,32,32))
        self.assertEqual(a.stage_descriptors(torch.tensor([[65.,1.,.6]]),normalized=False)[0,-1].tolist(),[65.,1.,torch.tensor(.6).item(),1.,3.])

    def test_zero_slot_after_normalization(self):
        x=torch.ones(2,1664);n={'mean':torch.full((1664,),.5),'std':torch.ones(1664)}
        y=normalize(x,n,True);self.assertTrue(torch.all(y[:,:640]==.5));self.assertTrue(torch.all(y[:,640:]==0))

    def test_search(self):
        result=search_tests()
        self.assertTrue(result['passed'])
        self.assertEqual(result['unique_per_task'],66)
        self.assertEqual(result['local_count'],30)

if __name__=='__main__':unittest.main()
