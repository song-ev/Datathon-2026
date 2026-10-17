"""Queue supervision tests: no GPU, no real videos, no five-minute sleeps."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from forte import app as api


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.patch=patch.object(api,'RUNTIME',Path(self.tmp.name));self.patch.start()
        api.pending.clear();api.jobs.clear();api.deadlines.clear();api.stopping.clear()
        api.state.update(ready=True,busy=False)

    def tearDown(self):
        api.pending.clear();api.jobs.clear();api.deadlines.clear();api.stopping.clear()
        self.patch.stop();self.tmp.cleanup()

    def add(self,jid,seconds=100):
        api.jobs[jid]={'job_id':jid,'status':'processing'}
        api.deadlines[jid]=time.monotonic()+seconds
        api.pending.append((jid,Path(jid+'.mov')))

    def test_fifo_and_error_do_not_block_next(self):
        for jid in ('a','b','c'):self.add(jid)
        order=[]
        def execute(jid,path):
            order.append(jid)
            api.jobs[jid]['status']='failed' if jid=='a' else 'done'
            api.state['busy']=False
            if jid=='c':api.stopping.set()
        with patch.object(api,'launch_worker'),patch.object(api,'execute_job',side_effect=execute):
            api.dispatch_loop()
        self.assertEqual(order,['a','b','c'])
        self.assertEqual(api.jobs['c']['status'],'done')

    def test_waiting_deadline_expires_while_gpu_busy(self):
        self.add('a',-1);self.add('b',60)
        api.state['busy']=True
        with api.lock:api.expire_pending()
        self.assertEqual(api.jobs['a']['status'],'failed')
        self.assertEqual(api.jobs['a']['error']['code'],'TimeoutError')
        self.assertEqual([p[0] for p in api.pending],['b'])
        self.assertTrue(api.state['busy'])

    def test_worker_receives_remaining_not_fresh_budget(self):
        self.add('a',25);api.pending.clear()
        pipe=Mock();pipe.poll.return_value=True
        pipe.recv.return_value=('result',{'verdict':'no_violation','violations':[]})
        proc=Mock();proc.is_alive.return_value=True
        with patch.object(api,'connection',pipe),patch.object(api,'process',proc):
            api.execute_job('a',Path('a.mov'))
        budget=pipe.send.call_args.args[0][1]
        self.assertTrue(0<budget<=23)
        self.assertEqual(api.jobs['a']['status'],'done')
        self.assertFalse(api.state['busy'])

    def test_expired_active_job_is_not_sent_to_gpu(self):
        self.add('a',-1);api.pending.clear()
        pipe=Mock();proc=Mock();proc.is_alive.return_value=True
        with patch.object(api,'connection',pipe),patch.object(api,'process',proc),patch.object(api,'launch_worker'):
            api.execute_job('a',Path('a.mov'))
        pipe.send.assert_not_called()
        self.assertEqual(api.jobs['a']['status'],'failed')
        proc.terminate.assert_called_once()

    def test_model_error_releases_gpu_for_next_job(self):
        self.add('a');api.pending.clear();api.state['busy']=True
        pipe=Mock();pipe.poll.return_value=True
        pipe.recv.return_value=('error',{'code':'UnsupportedScene','message':'unsupported'})
        proc=Mock();proc.is_alive.return_value=True
        with patch.object(api,'connection',pipe),patch.object(api,'process',proc):
            api.execute_job('a',Path('a.mov'))
        self.assertEqual(api.jobs['a']['status'],'failed')
        self.assertFalse(api.state['busy'])
        proc.terminate.assert_not_called()
