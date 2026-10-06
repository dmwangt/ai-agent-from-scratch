"""Deterministic external-service doubles for notebook regression tests only.

No Agent/LlmClient/tool implementation is replaced. Actual API payloads, model
availability, credentials, remote MCP processes and E2B connectivity need live tests.
"""
import ast
import contextlib
import io
import json
import os
import re
import socket
import tempfile
import shlex
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
for key in ('OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'TAVILY_API_KEY', 'HF_TOKEN', 'E2B_API_KEY'):
    os.environ[key] = 'offline-notebook-test'
import dotenv
dotenv.load_dotenv = lambda *a, **kw: False

import litellm
from litellm import ModelResponse


def sample(schema):
    if '$ref' in schema:
        raise AssertionError(f'Unresolved schema reference: {schema}')
    if 'enum' in schema: return schema['enum'][0]
    t = schema.get('type')
    if t == 'object': return {k: sample(v) for k,v in schema.get('properties', {}).items()}
    if t == 'array': return [sample(schema.get('items', {'type':'string'}))]
    if t in ('integer', 'number'): return 1
    if t == 'boolean': return True
    return '4'


SERVICE_EVENTS = []

def sandbox_step(text, messages):
    previous = [m for m in messages if m['role'] == 'tool']
    for message in previous:
        value = str(message['content'])
        assert not value.startswith(('Error', 'Upload error', 'Command error', 'INTENTIONALLY')), value
        try:
            payload = json.loads(value)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            assert not payload.get('error'), payload
    if 'Fibonacci' in text:
        if not previous:
            return ('execute_python', {'code': 'a, b = 0, 1\nfor _ in range(100):\n    a, b = b, a + b\na'}), None
        value = json.loads(previous[-1]['content'])['text']
        assert value == '354224848179261915075', value
        return None, value
    if 'Wikipedia' in text:
        if not previous:
            code = "revisions = get_wikipedia_revisions('2023-08-01', '2023-07-01T00:00:00Z', output='content')\nsum(page.count('twitter.com') for page in revisions)"
            return ('execute_python', {'code': code}), None
        value = json.loads(previous[-1]['content'])['text']
        assert value == '2', value
        return None, value
    if 'The file is located at:' in text:
        if not previous:
            # Read the path after the fixed notebook prompt, including spaces.
            filename = text.split('The file is located at:', 1)[1].strip()
            return ('upload_file', {'local_path': filename, 'sandbox_path': '/home/user/sales.xlsx'}), None
        if len(previous) == 1:
            assert 'File uploaded' in previous[0]['content'], previous[0]
            return ('bash_tool', {'command': 'ls /home/user'}), None
        if len(previous) == 2:
            assert 'sales.xlsx' in previous[-1]['content'], previous[-1]
            code = "import pandas as pd\ndf = pd.read_excel('/home/user/sales.xlsx')\ntotals = df.groupby('city')['sales'].sum()\nf'{totals.idxmax()}: {totals.max()}'"
            return ('execute_python', {'code': code}), None
        value = json.loads(previous[-1]['content'])['text']
        assert value == 'Wharvton: 250', value
        return None, value
    raise AssertionError(f'No scripted sandbox scenario for {text!r}')

def completion(*args, messages, tools=None, response_format=None, **kwargs):
    messages = [m.model_dump() if hasattr(m, 'model_dump') else m for m in messages]
    text = ' '.join(str(m.get('content') or '') for m in messages if m['role']=='user')
    defs = {t['function']['name']:t['function'] for t in (tools or [])}
    tc=None;content='4'
    if 'execute_python' in defs:
        tc, content = sandbox_step(text, messages)
    elif response_format:
        content=json.dumps(sample(response_format.model_json_schema()))
    elif 'final_answer' in defs:
        tc=('final_answer', sample(defs['final_answer']['parameters']))
    elif 'delete_file' in defs and not any(m['role']=='tool' for m in messages):
        filename = text.split('Please delete the file at ', 1)[1]
        tc=('delete_file', {'filename':filename})
    elif 'calculator' in defs and not any(m['role']=='tool' for m in messages):
        params=defs['calculator']['parameters'].get('properties',{})
        if 'expression' in params: tc=('calculator', {'expression':'7 * 8'})
        elif '1234' in text: tc=('calculator', {'operator':'multiply','first_number':1234,'second_number':5678})
    elif any(m['role']=='tool' for m in messages):
        content=str(next(m['content'] for m in reversed(messages) if m['role']=='tool'))
    msg={'role':'assistant','content':content}
    if tc:
        name, arguments=tc
        SERVICE_EVENTS.append({'kind':'tool_request','name':name})
        msg.update(content=None,tool_calls=[{'id':'call_offline','type':'function','function':{'name':name,'arguments':json.dumps(arguments)}}])
    return ModelResponse(choices=[{'index':0,'finish_reason':'tool_calls' if tc else 'stop','message':msg}],usage={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2})

async def acompletion(*args, **kwargs): return completion(*args, **kwargs)
litellm.completion=completion
litellm.acompletion=acompletion

# Keep the real SDK classes and Pydantic response models; intercept requests.
import openai
from openai.types import CreateEmbeddingResponse
from openai.resources.chat.completions import Completions
from openai.resources.embeddings import Embeddings
Completions.create=lambda self, **kw: completion(**kw)
def embedding(self, input, **kwargs):
    texts=[input] if isinstance(input,str) else input
    assert texts, 'Do not submit an empty embedding batch'
    return CreateEmbeddingResponse(model='offline',object='list',usage={'prompt_tokens':1,'total_tokens':1},data=[
        {'index':i,'object':'embedding','embedding':[float(len(t)%7+1),float(sum(t.encode())%11+1),1.0]} for i,t in enumerate(texts)])
Embeddings.create=embedding
from google.genai.models import Models
from google.genai.types import EmbedContentResponse, ContentEmbedding
def gemini_embedding(self, *, contents, model, **kwargs):
    texts = [contents] if isinstance(contents, str) else contents
    assert texts, 'Do not submit an empty embedding batch'
    return EmbedContentResponse(embeddings=[
        ContentEmbedding(values=[float(len(t)%7+1), float(sum(t.encode())%11+1), 1.0])
        for t in texts
    ])
Models.embed_content = gemini_embedding
import anthropic
from anthropic.resources.messages import Messages
Messages.create=lambda self, **kw: NS(content=[NS(type='text',text='4')])

from tavily import TavilyClient
TavilyClient.search=lambda self,*a,**kw: {'results':[{'title':'Offline fixture','content':'Quantum computing research. '*100,'raw_content':'Quantum computing research. '*100,'url':'https://example.invalid/fixture'}]}

import datasets
import huggingface_hub
fixture_rows=[{'task_id':str(i),'Question':'What is 2 + 2?','Final answer':'4','file_name':f'fixture-{i%2}.zip'} for i in range(30)]
datasets.load_dataset=lambda *a,**kw: datasets.Dataset.from_list(fixture_rows)
def download(*args, local_dir, **kwargs):
    import zipfile
    p=Path(local_dir)/'2023/validation';p.mkdir(parents=True,exist_ok=True)
    for i in range(2):
        with zipfile.ZipFile(p/f'fixture-{i}.zip','w') as z:z.writestr('sample.txt','Offline test data\n')
    return str(p)
huggingface_hub.snapshot_download=download

# Tokenizer download is external. Token estimates are not tested by this double.
import tiktoken
tiktoken.encoding_for_model=lambda *a: NS(encode=lambda s:list(s.encode()))
tiktoken.get_encoding=tiktoken.encoding_for_model

# Mock only MCP transport/session boundaries, preserving native protocol models.
import mcp
import mcp.client.stdio
from mcp.types import ListToolsResult, Tool, CallToolResult, TextContent
@contextlib.asynccontextmanager
async def stdio(*args,**kwargs): yield (None,None)
class Session:
    def __init__(self,*a,**kw):pass
    async def __aenter__(self):return self
    async def __aexit__(self,*a):pass
    async def initialize(self):pass
    async def list_tools(self):return ListToolsResult(tools=[Tool(name='search_web',description='Offline search',inputSchema={'type':'object','properties':{'query':{'type':'string'}},'required':['query']})])
    async def call_tool(self,*a,**kw):return CallToolResult(content=[TextContent(type='text',text='Offline search result')])
mcp.ClientSession=Session
mcp.client.stdio.stdio_client=stdio

# Wikipedia network boundary, used by the actual registered notebook function.
import requests
def wikipedia_get(url, params=None, **kwargs):
    assert url == 'https://en.wikipedia.org/w/api.php', url
    assert params['prop'] == 'revisions'
    SERVICE_EVENTS.append({'kind':'wikipedia_request'})
    return NS(json=lambda: {'query': {'pages': {'1': {'revisions': [{'*': 'twitter.com/a and twitter.com/b'}]}}}})
requests.get = wikipedia_get

# Execute the fixed notebook's sandbox snippets locally, not model-generated code.
# No credentials or network are available to this test process.
import e2b_code_interpreter
class Sandbox:
    def __init__(self):
        self.sandbox_id='offline-sandbox';self.scope={};self.closed=False
        self.directory = tempfile.TemporaryDirectory(prefix='offline-sandbox-')
        self.files=NS(write=self.write_file)
        self.commands=NS(run=self.run_command)
    def local_path(self, remote):
        assert remote.startswith('/home/user/'), remote
        relative = Path(remote.removeprefix('/home/user/'))
        assert '..' not in relative.parts
        return Path(self.directory.name) / relative
    def write_file(self, remote, data):
        assert not self.closed
        if hasattr(data, 'read'): data = data.read()
        if isinstance(data, str): data = data.encode()
        destination=self.local_path(remote)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
        SERVICE_EVENTS.append({'kind':'upload','path':remote,'bytes':len(data)})
    def run_command(self, command):
        assert not self.closed
        assert shlex.split(command) == ['ls', '/home/user'], command
        SERVICE_EVENTS.append({'kind':'command','command':command})
        return NS(stdout='\n'.join(sorted(p.name for p in Path(self.directory.name).iterdir())),stderr='',exit_code=0)
    @classmethod
    def create(cls,**kwargs):return cls()
    def set_timeout(self,*a):pass
    def kill(self):
        self.closed=True
        self.directory.cleanup()
    def run_code(self,code):
        assert not self.closed, 'Sandbox reused after kill()'
        output=io.StringIO();value=None;error=None
        try:
            tree=ast.parse(code)
            sandbox=self
            class MapPaths(ast.NodeTransformer):
                def visit_Constant(self,node):
                    if isinstance(node.value,str) and node.value.startswith('/home/user/'):
                        return ast.copy_location(ast.Constant(str(sandbox.local_path(node.value))),node)
                    return node
            tree=ast.fix_missing_locations(MapPaths().visit(tree))
            with contextlib.redirect_stdout(output):
                if tree.body and isinstance(tree.body[-1],ast.Expr):
                    exec(compile(ast.Module(tree.body[:-1],type_ignores=[]),'<sandbox>','exec'),self.scope)
                    value=eval(compile(ast.Expression(tree.body[-1].value),'<sandbox>','eval'),self.scope)
                else:exec(compile(tree,'<sandbox>','exec'),self.scope)
        except Exception as exc:error=NS(name=type(exc).__name__,value=str(exc))
        SERVICE_EVENTS.append({'kind':'code','text':str(value),'error':str(error) if error else None})
        return NS(error=error,logs=NS(stdout=output.getvalue().splitlines()),results=[NS(text=str(value))],to_json=lambda:{'text':str(value),'error':str(error) if error else None})
e2b_code_interpreter.Sandbox=Sandbox

# Fail unexpected external traffic while allowing the Jupyter kernel's localhost sockets.
_original_connect=socket.socket.connect
def guarded_connect(self,address):
    if isinstance(address,tuple) and address[0] not in ('127.0.0.1','localhost','::1'):
        raise AssertionError(f'Unexpected network request in offline test: {address}')
    return _original_connect(self,address)
socket.socket.connect=guarded_connect
print('OFFLINE TEST: provider responses, downloads, MCP transport and E2B are simulated.')
