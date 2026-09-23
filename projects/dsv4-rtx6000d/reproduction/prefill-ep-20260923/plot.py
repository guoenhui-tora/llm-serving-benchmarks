#!/usr/bin/env python3
"""Render the archived single-round curves with Matplotlib; no inference."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
data=json.loads(a.data.read_text());fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
colors={'on':'#0072B2','off':'#D55E00'}
for mode in ['on','off']:
 rows=[r for r in data['cells'] if r['mode']==mode];x=[r['concurrency'] for r in rows];color=colors[mode]
 axes[0,0].plot(x,[r['request_per_s'] for r in rows],'o-',color=color,label='EP '+mode)
 axes[0,1].plot(x,[r['mean_ttft_s'] for r in rows],'o-',color=color,label='EP '+mode+' mean')
 axes[0,1].plot(x,[r['p95_ttft_s'] for r in rows],'--',color=color,label='EP '+mode+' P95')
 axes[1,0].plot(x,[r['mean_queue_s'] for r in rows],'o-',color=color,label='EP '+mode)
 axes[1,1].plot(x,[r['mean_prefill_s'] for r in rows],'o-',color=color,label='EP '+mode)
for ax,title,ylabel in zip(axes.flat,['HTTP supply','Time to first token','Scheduler queue','Engine prefill stage'],['Requests / s','Seconds','Mean seconds','Mean seconds']):
 ax.set(title=title,xlabel='Client concurrency (entire four-GPU service)',ylabel=ylabel)
 ax.set_xticks([8,16,24,32,40,48,56,64]);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.axvline(16,color='#666666',linewidth=.8,alpha=.5);ax.legend(fontsize=8)
axes[0,0].set_ylim(0,1.35)
fig.suptitle('DSV4 16,384 input / 1 output | TP2 x DP2 | fixed S32, budget 16K, K5',fontsize=13)
fig.text(.5,-.02,'One 256-request round per point after 64-request warmup. EP mode is confounded with GPU/CPU placement.\nHTTP and engine times are not pure GPU timing; no KV transfer. Off C56 partly and C64 entirely after on completed.',ha='center',fontsize=9)
fig.savefig(a.output,dpi=180,bbox_inches='tight');fig.savefig(a.output.with_suffix('.svg'),bbox_inches='tight')
