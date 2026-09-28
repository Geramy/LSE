#!/usr/bin/env python3
"""Bounded native Chat Completions checks against an already running server."""
import argparse,json,pathlib,time,urllib.request
parser=argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:8080')
parser.add_argument('--model', default='qwen38-q4')
parser.add_argument('--output', type=pathlib.Path, required=True)
args=parser.parse_args()
base=args.base_url.rstrip('/')
out=args.output
out.mkdir(parents=True,exist_ok=True)
def req(name,body):
 body={'model':args.model,'temperature':0,**body}
 (out/(name+'-request.json')).write_text(json.dumps(body,indent=2))
 start=time.monotonic()
 request=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=300) as r: raw=r.read().decode()
 (out/(name+'-response.txt')).write_text(raw)
 result=[json.loads(x[6:]) for x in raw.splitlines() if x.startswith('data: ') and x!='data: [DONE]'] if body.get('stream') else json.loads(raw)
 print(name,round(time.monotonic()-start,2),json.dumps(result if not body.get('stream') else {'chunks':len(result),'tail':result[-2:]})[:1600],flush=True)
 return result
direct=req('direct',{'messages':[{'role':'developer','content':'Follow the user instruction exactly.'},{'role':'user','content':'Reply with exactly READY.'}],'enable_thinking':False,'max_completion_tokens':16})
assert 'READY' in direct['choices'][0]['message']['content'] and not direct['choices'][0]['message'].get('reasoning_content')
trunc=req('thinking-truncated',{'messages':[{'role':'user','content':'Work through 17 multiplied by 23.'}],'reasoning_effort':'low','max_tokens':8,'stream':True,'stream_options':{'include_usage':True}})
assert any(x.get('choices') and x['choices'][0]['delta'].get('reasoning_content') for x in trunc)
assert not any(x.get('choices') and x['choices'][0]['delta'].get('content') for x in trunc)
assert trunc[-2]['choices'][0]['finish_reason']=='length' and trunc[-1]['choices']==[]
think=req('thinking-complete',{'messages':[{'role':'user','content':'What is 2 plus 3? Think briefly, then give only the answer.'}],'reasoning_effort':'low','max_tokens':256})
assert think['choices'][0]['message'].get('reasoning_content') and '5' in think['choices'][0]['message']['content'],think
tool={'type':'function','function':{'name':'lookup','description':'Get the secret value for a key.','parameters':{'type':'object','properties':{'key':{'type':'string'}},'required':['key']}}}
messages=[{'role':'user','content':'Use lookup to get the secret value for key alpha. Do not guess the value.'}]
call=req('tool-nonstream',{'messages':messages,'tools':[tool],'tool_choice':{'type':'function','function':{'name':'lookup'}},'thinking':False,'max_tokens':160})
m=call['choices'][0]['message'];tc=m['tool_calls'][0]
assert call['choices'][0]['finish_reason']=='tool_calls' and tc['type']=='function' and tc['function']['name']=='lookup' and json.loads(tc['function']['arguments'])=={'key':'alpha'}
result=req('tool-result',{'messages':messages+[m,{'role':'tool','tool_call_id':tc['id'],'content':'{"value":"sapphire-517"}'}],'tools':[tool],'thinking':False,'max_tokens':64})
assert 'sapphire-517' in result['choices'][0]['message']['content']
stream=req('tool-stream',{'messages':messages,'tools':[tool],'tool_choice':'required','thinking':False,'max_tokens':160,'stream':True,'stream_options':{'include_usage':True}})
calls=[c for x in stream if x.get('choices') for c in x['choices'][0]['delta'].get('tool_calls',[])]
assert len(calls)==1 and calls[0]['index']==0 and calls[0]['id'] and json.loads(calls[0]['function']['arguments'])=={'key':'alpha'}
assert stream[-2]['choices'][0]['finish_reason']=='tool_calls' and stream[-1]['choices']==[] and stream[-1]['usage']['completion_tokens']>0
none=req('tool-none',{'messages':[{'role':'user','content':'Reply with exactly READY.'}],'tools':[tool],'tool_choice':'none','chat_template_kwargs':{'enable_thinking':False},'max_tokens':16})
assert 'READY' in none['choices'][0]['message']['content'] and 'tool_calls' not in none['choices'][0]['message']
print('RAW HTTP CHECKS PASS',flush=True)
