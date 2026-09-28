#!/usr/bin/env python3
"""Exercise the installed pi read tool against an already running LSE server."""
import argparse,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser()
p.add_argument('--base-url',default='http://127.0.0.1:8080/v1')
p.add_argument('--model',default='qwen38-q4')
p.add_argument('--context-window',type=int,default=32768)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--pi',default='pi')
a=p.parse_args()
if a.context_window<=4096:p.error('pi reserves 4096 context tokens; use a larger matching server window')
out=a.output.resolve()
out.mkdir(parents=True,exist_ok=True)
agent=out/'pi-agent'
agent.mkdir(exist_ok=True)
config={'providers':{'lse-check':{'baseUrl':a.base_url,'api':'openai-completions','apiKey':'local-test','compat':{'supportsStrictMode':False,'supportsStore':False,'thinkingFormat':'qwen'},'models':[{'id':a.model,'reasoning':True,'contextWindow':a.context_window,'maxTokens':384,'samplingParams':{'temperature':0}}]}}}
(agent/'models.json').write_text(json.dumps(config,indent=2))
fixture=out/'pi-fixture.txt'
fixture.write_text('The verification word is marigold-826.\n')
env={**os.environ,'PI_CODING_AGENT_DIR':str(agent)}
for mode in ['off','low']:
    command=[a.pi,'--provider','lse-check','--model',a.model,'--thinking',mode,'--mode','json','--print','--no-session','--no-extensions','--no-skills','--no-prompt-templates','--no-themes','--tools','read','--system-prompt','You are testing a local text file reader. Use the read tool when asked. After reading, give only the verification word. Keep any thinking brief.',f'Read {fixture} using the read tool and return its verification word.']
    start=time.monotonic()
    done=subprocess.run(command,cwd=out,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
    (out/f'pi-{mode}.jsonl').write_text(done.stdout)
    (out/f'pi-{mode}.stderr').write_text(done.stderr)
    assert done.returncode==0,done.stderr
    events=[json.loads(line) for line in done.stdout.splitlines() if line.startswith('{')]
    assert any(e.get('type')=='tool_execution_end' and e.get('toolName')=='read' and not e.get('isError') for e in events),done.stdout[-3000:]
    assistants=[e['message'] for e in events if e.get('type')=='message_end' and e['message'].get('role')=='assistant']
    assert assistants and assistants[-1].get('stopReason')=='stop',assistants
    answer=''.join(c.get('text','') for c in assistants[-1]['content'] if c['type']=='text')
    assert answer.strip()=='marigold-826',answer
    if mode=='low':assert any(c['type']=='thinking' and c.get('thinking') for m in assistants for c in m['content'])
    print(json.dumps({'thinking':mode,'result':'pass','elapsed_seconds':time.monotonic()-start,'answer':answer}),flush=True)
