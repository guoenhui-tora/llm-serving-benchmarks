from __future__ import annotations

import ast
import asyncio
import copy
import inspect
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serving_bench import deployment
from serving_bench.clients import synchronized, vllm_bench
from serving_bench.common import BenchError, read_json, write_json
from serving_bench.config import resolve, validate_replica_targets

ROOT = Path(__file__).resolve().parents[1]


# Real Python source used to check instrumentation without Docker or GPUs.
async def benchmark(input_requests):
    benchmark_start_time = time.perf_counter()
    outputs = input_requests
    benchmark_duration = time.perf_counter() - benchmark_start_time
    return outputs, benchmark_duration


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = resolve(ROOT/'tests/fixtures/configs/campaigns/46-glm52-vllm-smoke.yaml')['cases'][0]
        self.case['target']['binding']={'server':{'cpus':'0-7','mems':'0'},'client':{'cpus':'8-11','mems':'0'}}
        self.case['recipe']['options']['tensor-parallel-size']=4
        members=[]
        for i in range(2):
            member=copy.deepcopy(self.case)
            member['target'].update(gpus=list(range(4*i,4*i+4)),port=31248+i)
            member['target']['binding']={'server':{'cpus':f'{4*i}-{4*i+3}','mems':'0'},'client':{'cpus':f'{8+2*i}-{9+2*i}','mems':'0'}}
            members.append(member)
        self.case['replicas']=members

    def test_overlap_or_incomplete_partition_rejected(self):
        target=self.case['target']; members=[m['target'] for m in self.case['replicas']]
        for change in ['gpu','port','cpu','budget']:
            values=copy.deepcopy(members)
            if change=='gpu':values[1]['gpus'][0]=0
            if change=='port':values[1]['port']=values[0]['port']
            if change=='cpu':values[1]['binding']['server']['cpus']='0-15'
            if change=='budget':values[1]['binding']['server']['cpus']='4-6'
            with self.subTest(change=change),self.assertRaises(BenchError):
                validate_replica_targets(target,values)

    def test_clock_hook_uses_native_timing_and_disjoint_dataset(self):
        coordination=self.root/'coord';coordination.mkdir()
        write_json(coordination/'release.json',{'start_ns':time.monotonic_ns()})
        collected=[]
        for index in range(2):
            output=self.root/str(index);output.mkdir()
            serve=SimpleNamespace(benchmark=benchmark,calculate_metrics=lambda:None)
            synchronized.install(serve,coordination,output,index,2,2)
            result,duration=asyncio.run(serve.benchmark(input_requests=list(range(8))))
            collected.extend(result)
            self.assertEqual(result,list(range(index,8,2)))
            self.assertEqual(read_json(output/'benchmark-window.json')['duration_s'],duration)
        self.assertEqual(sorted(collected),list(range(8)))

    def test_clock_layout_changes_fail_closed(self):
        source=inspect.getsource(benchmark)
        for bad in [source.replace('time.perf_counter()', 'time.time()'),source.replace('benchmark_start_time =','changed_name =')]:
            with self.assertRaises(RuntimeError):synchronized.instrument(bad)

    def make_results(self):
        directories=[]
        for index in range(2):
            p=self.root/f'replica-{index}';p.mkdir();directories.append(p)
            duration=2+index
            write_json(p/'benchmark-window.json',{'start_s':100+index*.01,'end_s':100+index*.01+duration})
            write_json(p/'metrics.json',{'completed':2,'metrics':{'duration':duration,'total_input_tokens':16,'total_output_tokens':8,'output_throughput':8/duration}})
            rows=[{'success':True,'input_tokens':8,'output_tokens':4,'ttft_s':t,'latency_s':t+.3,'itl_s':[.1,.1,.1]} for t in ([.1,.2] if index==0 else [.3,1])]
            write_json(p/'requests.json',rows)
        workload={'dataset':{'input_tokens':8,'output_tokens':4,'range_ratio':0},'sampling':{'ignore_eos':True}}
        return directories,workload

    def test_union_window_and_pooled_percentiles_not_sum_or_mean(self):
        directories,w=self.make_results();normalized,window=deployment.aggregate(directories,4,w)
        m=normalized['metrics']
        self.assertAlmostEqual(m['duration'],3.01)
        self.assertAlmostEqual(m['output_throughput'],16/3.01)
        self.assertAlmostEqual(m['p95_ttft_ms'],895)
        self.assertAlmostEqual(m['mean_tpot_ms'],100)
        self.assertAlmostEqual(sum(r['output_throughput_common_window'] for r in window['replicas']),m['output_throughput'])

    def test_missing_tokens_and_unsynchronized_windows_fail(self):
        directories,w=self.make_results()
        rows=read_json(directories[1]/'requests.json');rows[0]['output_tokens']=3;write_json(directories[1]/'requests.json',rows)
        with self.assertRaisesRegex(BenchError,'output token'):deployment.aggregate(directories,4,w)
        rows[0]['output_tokens']=4;write_json(directories[1]/'requests.json',rows)
        write_json(directories[1]/'benchmark-window.json',{'start_s':101,'end_s':104})
        with self.assertRaisesRegex(BenchError,'skew'):deployment.aggregate(directories,4,w)

    def test_group_phase_propagates_either_replica_compilation_events(self):
        from serving_bench import runner
        case=self.case; workload=case['workloads'][0]
        sessions=[(m,f'server-{i}',self.root) for i,m in enumerate(case['replicas'])]
        def fake_phase(member, workload, concurrency, count, directory, server, owner, facts):
            directory.mkdir(parents=True)
            sync=member['_client_sync'];coord=Path(sync['directory']);i=sync['index']
            write_json(coord/f'ready-{i}.json',{})
            while not (coord/'release.json').exists():time.sleep(.001)
            return {},(['TileLang begins to compile kernel'] if i==1 else [])
        with patch.object(runner,'phase',side_effect=fake_phase),patch.object(deployment,'aggregate',return_value=({'completed':128},{})):
            _,events=deployment.phase(case,workload,32,128,self.root/'phase',sessions,'test',{'replicas':[{},{}]})
        self.assertEqual(events,[{'replica':1,'event':'TileLang begins to compile kernel'}])

    def test_protocol_dual_replica_events_and_equal_share(self):
        from serving_bench import runner
        from serving_bench.protocols import collect
        workload=copy.deepcopy(self.case['workloads'][0])
        workload['traffic']['requests']=128
        workload['measurement']=dict(protocol='jit_clean',repetitions=3,max_rounds=5,timeout_s=30,budget_s=100)
        sessions=[(m,f'server-{i}',self.root) for i,m in enumerate(self.case['replicas'])]
        calls=[]
        def fake_phase(member, w, concurrency, count, directory, server, owner, facts):
            directory.mkdir(parents=True)
            sync=member['_client_sync'];coord=Path(sync['directory']);i=sync['index']
            calls.append((concurrency,count,i))
            write_json(coord/f'ready-{i}.json',{})
            while not (coord/'release.json').exists():time.sleep(.001)
            event=i==1 and directory.parent.name=='measurement-01'
            return {},(['JIT'] if event else [])
        def measure(count,where,timeout):
            return deployment.phase(self.case,workload,32,count,where,sessions,'owned',{'replicas':[{},{}]})
        metric={'completed':128,'metrics':{'output_throughput':100,'mean_ttft_ms':20,'mean_tpot_ms':5}}
        with patch.object(runner,'phase',side_effect=fake_phase),patch.object(deployment,'aggregate',return_value=(metric,{})):
            trials,state=collect(workload,32,self.root/'block',measure,lambda _:None,lambda _:None)
        self.assertEqual(state['selected_rounds'],[2,3,4])
        self.assertEqual(len(calls),8)
        self.assertTrue(all(c[:2]==(16,64) for c in calls))
        self.assertEqual(len(trials),3)

    def test_group_phase_failure_aborts_peer_and_cleans_only_owned_clients(self):
        from serving_bench import runner
        sessions=[(m,f'server-{i}',self.root) for i,m in enumerate(self.case['replicas'])]
        def fail(member,*args):
            sync=member['_client_sync']
            if sync['index']==0:raise BenchError('failed client')
            while not (Path(sync['directory'])/'abort.json').exists():time.sleep(.001)
            raise BenchError('peer aborted')
        with patch.object(runner,'phase',side_effect=fail),patch.object(deployment.docker,'remove_owned') as remove:
            with self.assertRaisesRegex(BenchError,'failed client'):
                deployment.phase(self.case,self.case['workloads'][0],32,128,self.root/'phase',sessions,'owned',{'replicas':[{},{}]})
        self.assertEqual({call.args for call in remove.call_args_list},{('sb-owned-r0-client','owned-r0'),('sb-owned-r1-client','owned-r1')})

    def test_stop_cleans_each_owned_replica(self):
        import socket
        from serving_bench import runner
        write_json(self.root/'run.json',{'hostname':socket.gethostname(),'status':'RUNNING','owner':'owned'})
        write_json(self.root/'plan.json',{'cases':[self.case]})
        with patch.object(deployment.docker,'remove_owned') as remove:
            runner.stop(self.root)
        self.assertTrue((self.root/'stop-requested.json').exists())
        self.assertIn(('sb-owned-r0-server','owned-r0'),[c.args for c in remove.call_args_list])
        self.assertIn(('sb-owned-r1-client','owned-r1'),[c.args for c in remove.call_args_list])

    def test_export_preserves_node_and_replica_scopes(self):
        import shutil
        from scripts.export_samples import samples
        directories,w=self.make_results();node,window=deployment.aggregate(directories,4,w)
        phase=self.root/'trial/measurement-01';phase.mkdir(parents=True)
        for i,p in enumerate(directories):shutil.move(str(p),str(phase/f'replica-{i}'))
        write_json(phase/'benchmark-window.json',window)
        write_json(self.root/'run.json',{'owner':'one','cases':[{'path':'.'}]})
        write_json(self.root/'case.json',{'status':'PASS','trials':[{'path':'trial','measurement':'measurement-01',
            'concurrency':4,'repetition':1,'workload':'test','purpose':'performance','metrics':node}]})
        write_json(self.root/'resolved.json',self.case)
        write_json(self.root/'environment.json',{'hostname':'node-test'})
        rows=samples([self.root,self.root])
        self.assertEqual([r['scope'] for r in rows],['node','replica-0','replica-1'])
        self.assertAlmostEqual(rows[0]['output_tokens_per_s'],sum(r['output_tokens_per_s_common_window'] for r in rows[1:]))
        self.assertTrue(all(r['resource_status']=='review-required' for r in rows))
        self.assertTrue(all('source_path' not in r for r in rows))

    def test_client_command_partitions_global_dataset_before_measurement(self):
        case=copy.deepcopy(self.case['replicas'][0]);case['_client_sync']={'directory':str(self.root),'index':0,'replicas':2,'global_requests':128,'timeout_s':300}
        args=vllm_bench.command(case,case['workloads'][0],16,64,self.root,'test','owner')
        self.assertEqual(args[args.index('--num-prompts')+1],'128')
        self.assertEqual(args[args.index('--max-concurrency')+1],'16')
        self.assertIn('/synchronized.py',args[args.index('-c')+1])
