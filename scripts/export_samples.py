#!/usr/bin/env python3
"""Export portable accepted node/replica samples without local source paths."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from serving_bench.common import BenchError, read_json
from serving_bench.results.report import artifact

METRICS = {'duration': 'duration_s', 'output_throughput': 'output_tokens_per_s',
           'request_throughput': 'requests_per_s', 'total_input_tokens': 'actual_input_tokens',
           'total_output_tokens': 'actual_output_tokens'}
METRICS.update({f'{stat}_{metric}_ms':f'{stat}_{metric}_ms'
                for metric in ['ttft','tpot','itl','e2el'] for stat in ['mean','p95']})
FIELDS = ['sample_id','node','configuration','workload','purpose','scope','gpu_group','deployment_count','tp','pp','dp','ep',
          'global_capacity','prefill_tokens_per_rank','prefill_tokens_service','global_concurrency',
          'concurrency','repetition','successful_requests','failed_requests',*METRICS.values(),
          'output_tokens_per_s_common_window','start_skew_s','overlap_s','runner_gate','resource_status','acceptance_protocol','protocol_version','compilation_check','event_count','round']


def samples(roots, resource_status='review-required'):
    output=[];seen_owners=set();seen_ids=set()
    for root in roots:
        run=read_json(root/'run.json')
        if run['owner'] in seen_owners:
            continue
        seen_owners.add(run['owner'])
        for ref in run['cases']:
            where=artifact(root,ref['path'])
            if not (where/'case.json').exists():
                continue
            state=read_json(where/'case.json')
            if state['status'] not in {'PASS','PARTIAL'}:
                continue
            case=read_json(where/'resolved.json');facts=read_json(where/'environment.json')
            if case['runtime']['engine'] != 'vllm':
                raise BenchError('CSV topology exporter currently supports vLLM cases only')
            members=case.get('replicas',[case]);n=len(members);options=case['recipe']['options']
            tp=options.get('tensor-parallel-size',1);pp=options.get('pipeline-parallel-size',1);dp=options.get('data-parallel-size',1)
            for trial in state['trials']:
                if trial.get('protocol_status','PASS') != 'PASS':
                    continue
                if state['status']=='PARTIAL' and 'protocol_status' not in trial:
                    continue
                path=artifact(where,trial['path']);phase=artifact(path,trial['measurement'])
                window=read_json(phase/'benchmark-window.json') if 'replicas' in case else {}
                records=[('node',trial['metrics'],case['target']['gpus'],trial['concurrency'],None)]
                if 'replicas' in case:
                    for i,member in enumerate(members):
                        records.append((f'replica-{i}',read_json(phase/f'replica-{i}/metrics.json'),member['target']['gpus'],
                                        trial['concurrency']//n,window['replicas'][i]))
                for scope,result,gpus,concurrency,part in records:
                    identity=f"{facts['hostname']}-{case['id']}-{trial['workload']}-c{trial['concurrency']}-r{trial['repetition']:02d}-{scope}"
                    if identity in seen_ids:
                        raise BenchError('Duplicate sample identity across runs; do not silently merge repeated batches: '+identity)
                    seen_ids.add(identity)
                    row={'sample_id':identity,'node':facts['hostname'],'configuration':case['id'],'workload':trial['workload'],'purpose':trial['purpose'],'scope':scope,
                         'gpu_group':','.join(map(str,gpus)),'deployment_count':n,'tp':tp,'pp':pp,'dp':dp,
                         'ep':'on' if 'enable-expert-parallel' in case['recipe']['flags'] else 'off',
                         'global_capacity':n*dp*options.get('max-num-seqs',0),
                         'prefill_tokens_per_rank':options.get('max-num-batched-tokens'),
                         'prefill_tokens_service':n*dp*options.get('max-num-batched-tokens',0),
                         'global_concurrency':trial['concurrency'],'concurrency':concurrency,
                         'repetition':trial['repetition'],'successful_requests':result['completed'],
                         'failed_requests':0 if result.get('failure_count_inferred') else result.get('failed'),
                         'runner_gate':'QUICK_COMPLETE' if trial.get('protocol')=='quick' else 'PASS',
                         'acceptance_protocol':trial.get('protocol','legacy'),'protocol_version':trial.get('protocol_version',1),
                         'compilation_check':trial.get('compilation_check','NO_KNOWN_EVENTS'),
                         'event_count':trial.get('event_count',0),'round':trial.get('round',trial['repetition']),
                         'resource_status':resource_status,
                         'start_skew_s':window.get('start_skew_s'),'overlap_s':window.get('overlap_s')}
                    row.update({column:result['metrics'].get(metric) for metric,column in METRICS.items()})
                    row['output_tokens_per_s_common_window']=part['output_throughput_common_window'] if part else result['metrics']['output_throughput']
                    output.append(row)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resource-status',default='review-required',choices=['review-required','accepted-background','resource-affected'])
    args=parser.parse_args()
    try:
        rows=samples([p.resolve() for p in args.runs],args.resource_status)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
        print(f'Exported {len(rows)} rows; resource qualification must be reviewed separately.')
    except (BenchError,OSError,ValueError) as exc:
        parser.exit(1,f'ERROR: {exc}\n')


if __name__=='__main__':main()
