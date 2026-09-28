from pathlib import Path
import json, re, statistics, sys
prefix = Path(sys.argv[1])
p = prefix.parent
records = [json.loads(line) for line in Path(str(prefix)+'.jsonl').read_text().splitlines()]
events = [r for r in records if r['record_type'] == 'dispatch_event']
assert len(events) == 576, len(events)
assert all(r['valid'] and r['duration_scale_available'] for r in events)
def stats(v):
    q = statistics.quantiles(v,n=100,method='inclusive')
    return dict(n=len(v),mean_ms=statistics.mean(v),median_ms=statistics.median(v),min_ms=min(v),p95_ms=q[94],max_ms=max(v))
def sums(v):
    return [(a['duration_ns']+b['duration_ns'])/1e6 for a,b in zip(v[::2],v[1::2])]
r = {}
for ci,name in enumerate(['q4_up_m4','q4_down_m4']):
    e=events[ci*288:(ci+1)*288]
    prep,panel=e[0]['key'],e[1]['key']
    partial,merge=e[64]['key'],e[65]['key']
    assert len({prep,panel,partial,merge})==4
    assert [v['key'] for v in e[:64]] == [prep,panel]*32
    assert [v['key'] for v in e[64:128]] == [partial,merge]*32
    blocks=[e[128:168],e[168:208],e[208:248],e[248:288]]
    assert all([v['key'] for v in blocks[i]] == ([prep,panel] if i in [0,3] else [partial,merge])*20 for i in range(4))
    ba=[stats(sums(blocks[i])) for i in [0,3]]
    ca=[stats(sums(blocks[i])) for i in [1,2]]
    baseline=stats(sums(blocks[0]+blocks[3])); candidate=stats(sums(blocks[1]+blocks[2]))
    wall={a:[float(v) for v in re.findall(r'timing case='+name+r' arm='+a+r'.*?per_call_ms=([0-9.]+)',Path(str(prefix)+'.log').read_text())] for a in ['panel','splitk4']}
    r[name]=dict(baseline=baseline,candidate=candidate,baseline_arms=ba,candidate_arms=ca,wall_ms_per_call=wall,gpu_sum_change_percent=(candidate['mean_ms']/baseline['mean_ms']-1)*100,wall_change_percent=(statistics.mean(wall['splitk4'])/statistics.mean(wall['panel'])-1)*100,both_cp_arms_clear=max(v['mean_ms'] for v in ca)<min(v['mean_ms'] for v in ba),warm_calls_per_arm=32,measured_calls_per_arm=40,clock_frequency_hz=100000000,host_clock_correlated=False)
Path(str(prefix)+'-cp-timings.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
