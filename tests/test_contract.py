import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from forte.engine import parse_json,evidence,probe,sample,RULES
from scripts.evaluate import split_records,metrics

class ContractTests(unittest.TestCase):
    def test_shell_entrypoints_use_linux_line_endings(self):
        for path in (Path(__file__).resolve().parents[1]/'scripts').glob('*.sh'):
            self.assertNotIn(b'\r',path.read_bytes(),str(path))

    def test_parse_fenced_json(self):
        self.assertEqual(parse_json('```json\n{"events":[]}\n```'),{'events':[]})

    def test_invalid_is_not_no_violation(self):
        for text in ['no problems','{}','{"events":null}']:
            with self.assertRaises(ValueError):parse_json(text)

    def test_duplicate_groups_never_cross_split(self):
        records=[]
        for label in ['нарушение','без нарушения']:
            for i in range(20):
                r={'sha256':label+str(i),'label':label};records.extend([r,r.copy()])
        dev,test=split_records(records)
        self.assertFalse({x[0] for x in dev}&{x[0] for x in test})
        self.assertEqual(len(dev),24);self.assertEqual(len(test),16)
        self.assertEqual(sum(g['label'] for _,g in test),8)

    def test_confusion_and_time_metrics(self):
        rows=[{'label':1,'times':[10],'candidates':[{'confidence':.9,'time_sec':11}]},
              {'label':0,'times':[],'candidates':[{'confidence':.7,'time_sec':5}]},
              {'label':1,'times':[4],'candidates':[]}]
        m=metrics(rows,.8)
        self.assertEqual((m['tp'],m['fp'],m['tn'],m['fn']),(1,0,1,1))
        self.assertEqual(m['positive_localization_recall_2s'],.5)

    def test_evidence_is_original_timestamped_jpeg(self):
        import base64,io
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.mov'
            with av.open(str(p),'w') as c:
                s=c.add_stream('libx264',rate=10);s.width=64;s.height=64;s.pix_fmt='yuv420p'
                for i in range(20):
                    a=np.full((64,64,3),i*10,dtype=np.uint8)
                    for packet in s.encode(av.VideoFrame.from_ndarray(a,format='rgb24')):c.mux(packet)
                for packet in s.encode():c.mux(packet)
            self.assertAlmostEqual(probe(p),2,places=1)
            self.assertEqual(len(sample(p)),2)
            result=evidence(p,[.3,1,1.5])
            self.assertEqual(len(result),3)
            for item,t in zip(result,[.3,1,1.5]):
                self.assertAlmostEqual(item['timestamp_sec'],t,places=2)
                im=Image.open(io.BytesIO(base64.b64decode(item['image_base64'])))
                self.assertEqual(im.format,'JPEG');self.assertEqual(im.size,(64,64))

if __name__=='__main__':unittest.main()
