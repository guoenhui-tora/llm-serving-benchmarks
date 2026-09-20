import copy
import unittest
from pathlib import Path
from serving_bench.config import resolve, validate_document
from serving_bench.common import BenchError
from serving_bench.executors.docker import server_command, OWNER_LABEL
from serving_bench.pd_pair import owned
from serving_bench.pd_proxy import prefill_request, transfer_params

class PDConfigTests(unittest.TestCase):
    def test_explicit_listener_devices_and_defaults(self):
        c=resolve(Path('tests/fixtures/configs/campaigns/46-glm52-vllm-smoke.yaml'))['cases'][0]
        args=server_command(c,'test','owner')
        self.assertEqual(args[args.index('--host')+1],'127.0.0.1')
        self.assertNotIn('--device',args)
        c['target'].update(listen_address='0.0.0.0',devices=['/dev/infiniband'])
        validate_document(c['target'],'target')
        args=server_command(c,'test','owner')
        self.assertEqual(args[args.index('--host')+1],'0.0.0.0')
        self.assertEqual(args[args.index('--device')+1],'/dev/infiniband')
        c['target']['devices']=['/dev/../etc']
        with self.assertRaises(BenchError):validate_document(c['target'],'target')

    def test_cleanup_requires_matching_owner(self):
        self.assertFalse(owned([], 'a'))
        self.assertFalse(owned([{'Config':{'Labels':{OWNER_LABEL:'b'}}}],'a'))
        self.assertTrue(owned([{'Config':{'Labels':{OWNER_LABEL:'a'}}}],'a'))

    def test_prefill_preserves_decode_body(self):
        original={'stream':True,'max_tokens':1024,'min_tokens':1024,'stream_options':{'include_usage':True},'prompt':'x'}
        before=copy.deepcopy(original);p=prefill_request(original)
        self.assertEqual(original,before);self.assertEqual(p['max_tokens'],1)
        self.assertFalse(p['stream']);self.assertNotIn('min_tokens',p)
        self.assertTrue(p['kv_transfer_params']['do_remote_decode'])

    def test_metadata_fails_closed(self):
        good=dict(do_remote_prefill=True,remote_engine_id='e',remote_request_id='r',remote_host='host',remote_port=5600,remote_block_ids=[[1,2]],transfer_mode='pull')
        self.assertEqual(transfer_params(good),good)
        for x in [None,{},dict(good,remote_block_ids=[]),dict(good,remote_block_ids=[[]]),dict(good,remote_host=None),dict(good,transfer_mode='push')]:
            with self.assertRaises(ValueError):transfer_params(x)

if __name__=='__main__':unittest.main()
