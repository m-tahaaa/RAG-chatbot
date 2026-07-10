import os, json, traceback
from dotenv import load_dotenv
load_dotenv()
from azure.identity import DefaultAzureCredential
import requests

endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
api_ver = os.getenv('AZURE_OPENAI_API_VERSION')
prompt = 'Hello from debug script. Answer briefly from context: test'
print('endpoint:', endpoint)
try:
    cred = DefaultAzureCredential()
    token = cred.get_token('https://cognitiveservices.azure.com/.default')
    print('got token, len=', len(token.token))
    headers = {'Authorization': f'Bearer {token.token}', 'Content-Type': 'application/json'}
    url = endpoint
    if api_ver:
        if '?' in url:
            url = url + '&api-version=' + api_ver
        else:
            url = url + '?api-version=' + api_ver
    body = {'input': prompt}
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    print('status', resp.status_code)
    print(resp.text[:4000])
except Exception as e:
    traceback.print_exc()
    print('err', str(e))
