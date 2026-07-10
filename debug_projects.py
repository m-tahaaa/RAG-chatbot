import traceback
import os
from dotenv import load_dotenv
load_dotenv()
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import VersionRefIndicator

endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
agent = os.getenv('AZURE_OPENAI_DEPLOYMENT')
api_ver = os.getenv('AZURE_OPENAI_API_VERSION')
print('endpoint:', endpoint)
print('agent:', agent)
print('api_ver:', api_ver)
try:
    cred = DefaultAzureCredential()
    with AIProjectClient(endpoint=endpoint, credential=cred, allow_preview=True) as client:
        print('client created')
        print('beta.agents dir:', [n for n in dir(client.beta.agents) if not n.startswith('_')])
        # Attempt to find a session creation method
        for name in dir(client.beta.agents):
            if 'session' in name.lower() or 'create' in name.lower():
                print('candidate method:', name)
        # Try to call any obvious method names if present
        if hasattr(client.beta.agents, 'begin_create_session'):
            sess = client.beta.agents.begin_create_session(agent_name=agent, isolation_key='pdf-rag')
            print('begin_create_session returned:', sess)
        elif hasattr(client.beta.agents, 'create_agent_session'):
            sess = client.beta.agents.create_agent_session(agent_name=agent, isolation_key='pdf-rag')
            print('create_agent_session returned:', sess)
        else:
            print('No session creation method found on beta.agents')
except Exception as e:
    print('EXCEPTION:')
    traceback.print_exc()
    print('str(e):', str(e))
