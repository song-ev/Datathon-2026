"""Leakage and input-contract checks for adapter training."""
import json,tempfile,unittest
from pathlib import Path
from scripts.train_lora import records,stats

class LoraProtocolTests(unittest.TestCase):
    def test_disjoint_video_groups(self):
        rows=[]
        for label,n in [('нарушение',21),('без нарушения',19)]:
            for i in range(n):
                row={'sha256':label+str(i),'label':label,'name':str(i)+'.mov'}
                rows.append(row)
                if i==0:rows.append(dict(row))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'data').mkdir();(root/'data/audit.json').write_text(json.dumps({'records':rows}),encoding='utf8')
            train,val,test,dev=records(root)
        sets=[{s for s,_ in part} for part in [train,val,test]]
        self.assertEqual([len(s) for s in sets],[18,6,16])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(sets[0]|sets[1],{s for s,_ in dev})
        self.assertEqual(sum(g['label'] for _,g in val),3)

    def test_recall_does_not_hide_false_positives(self):
        r=stats([1,1,0,0],[.9,.8,.7,.6])
        self.assertEqual(r['recall'],1)
        self.assertEqual(r['specificity'],0)
        self.assertEqual(r['accuracy'],.5)
        self.assertEqual(r['fp'],2)

    def test_prompt_does_not_include_file_identity(self):
        from forte.lora import inputs_for
        from PIL import Image
        class Processor:
            def apply_chat_template(self,messages,**kwargs):return messages
        frames=[(float(i),Image.new('RGB',(464,848))) for i in range(21)]
        messages=inputs_for(Processor(),'private_violation_label.mov',bounds=(4,16),frames=frames)
        text=' '.join(c.get('text','') for c in messages[0]['content'])
        self.assertNotIn('private_violation_label',text)
        self.assertIn('t=0.0s',text)
        self.assertIn('t=12.0s',text)
        self.assertEqual(sum(c['type']=='image' for c in messages[0]['content']),12)

if __name__=='__main__':unittest.main()
