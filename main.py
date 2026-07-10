import io
import os
import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
try:
	from azure.identity import DefaultAzureCredential
	from azure.ai.projects import AIProjectClient
	from azure.ai.projects.models import VersionRefIndicator
	_HAS_AZURE_PROJECTS = True
except Exception:
	_HAS_AZURE_PROJECTS = False
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
	SearchFieldDataType,
	SearchIndex,
	SearchableField,
	SimpleField,
)
import requests


load_dotenv()


app = FastAPI(
	title="PDF RAG Backend",
	description="Upload PDFs to Azure Blob Storage, index them in Azure AI Search, and answer questions from the indexed document only.",
	version="1.0.0",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
	app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class UploadResponse(BaseModel):
	document_id: str
	file_name: str
	title: str
	blob_name: str
	blob_url: str
	pages_indexed: int
	chunks_indexed: int
	indexed_at: datetime


class AskRequest(BaseModel):
	question: str = Field(..., min_length=1)
	top_k: int = Field(default=5, ge=1, le=10)


class SourceChunk(BaseModel):
	chunk_id: str
	page_number: int
	score: Optional[float] = None
	snippet: str


class AskResponse(BaseModel):
	document_id: str
	question: str
	answer: str
	sources: List[SourceChunk]
	retrieved_at: datetime


@dataclass(frozen=True)
class Settings:
	azure_search_endpoint: str
	azure_search_key: str
	azure_search_index: str
	azure_openai_endpoint: str
	azure_openai_api_key: str
	azure_openai_deployment: str
	azure_openai_api_version: str
	azure_openai_use_projects: bool = False
	max_chunk_chars: int = 1600
	chunk_overlap_chars: int = 200


def get_settings() -> Settings:
	return Settings(
		azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", "").strip(),
		azure_search_key=os.getenv("AZURE_SEARCH_KEY", "").strip(),
		azure_search_index=os.getenv("AZURE_SEARCH_INDEX", "pdf-documents").strip(),
		azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
		azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
		azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip(),
		azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-08-07").strip(),
		azure_openai_use_projects=os.getenv("AZURE_OPENAI_USE_PROJECTS", "false").lower() in ("1", "true", "yes"),
	)


SETTINGS = get_settings()
SEARCH_CREDENTIAL = AzureKeyCredential(SETTINGS.azure_search_key) if SETTINGS.azure_search_key else None
SEARCH_INDEX_CLIENT = (
	SearchIndexClient(endpoint=SETTINGS.azure_search_endpoint, credential=SEARCH_CREDENTIAL)
	if SETTINGS.azure_search_endpoint and SEARCH_CREDENTIAL
	else None
)
SEARCH_CLIENT = (
	SearchClient(
		endpoint=SETTINGS.azure_search_endpoint,
		index_name=SETTINGS.azure_search_index,
		credential=SEARCH_CREDENTIAL,
	)
	if SETTINGS.azure_search_endpoint and SEARCH_CREDENTIAL
	else None
)
def build_openai_client() -> Optional[Any]:
	if not (SETTINGS.azure_openai_api_key and SETTINGS.azure_openai_endpoint):
		return None

	endpoint = SETTINGS.azure_openai_endpoint.rstrip("/")
	if "services.ai.azure.com" in endpoint or "/api/projects/" in endpoint:
		return OpenAI(api_key=SETTINGS.azure_openai_api_key, base_url=f"{endpoint}/openai/v1")

	return AzureOpenAI(
		api_key=SETTINGS.azure_openai_api_key,
		azure_endpoint=endpoint,
		api_version=SETTINGS.azure_openai_api_version,
	)


AZURE_OPENAI_CLIENT = build_openai_client()


def require_config(value: Optional[str], message: str) -> str:
	if not value:
		raise HTTPException(status_code=503, detail=message)
	return value


def require_services() -> None:
	require_config(SETTINGS.azure_search_endpoint, "AZURE_SEARCH_ENDPOINT is required")
	require_config(SETTINGS.azure_search_key, "AZURE_SEARCH_KEY is required")
	require_config(SETTINGS.azure_search_index, "AZURE_SEARCH_INDEX is required")
	require_config(SETTINGS.azure_openai_endpoint, "AZURE_OPENAI_ENDPOINT is required")
	require_config(SETTINGS.azure_openai_api_key, "AZURE_OPENAI_API_KEY is required")
	require_config(SETTINGS.azure_openai_deployment, "AZURE_OPENAI_DEPLOYMENT is required")


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
	reader = PdfReader(io.BytesIO(pdf_bytes))
	pages: list[dict[str, Any]] = []
	for page_number, page in enumerate(reader.pages, start=1):
		text = (page.extract_text() or "").strip()
		if text:
			pages.append({"page_number": page_number, "text": text})
	return pages


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
	normalized = re.sub(r"\s+", " ", text).strip()
	if not normalized:
		return []

	chunks: list[str] = []
	start = 0
	while start < len(normalized):
		end = min(start + max_chars, len(normalized))
		chunk = normalized[start:end].strip()
		if chunk:
			chunks.append(chunk)
		if end >= len(normalized):
			break
		start = max(end - overlap_chars, start + 1)
	return chunks


def make_chunk_id(document_id: str, page_number: int, chunk_number: int) -> str:
	return f"{document_id}-p{page_number}-c{chunk_number}"


def ensure_search_index() -> None:
	require_services()
	assert SEARCH_INDEX_CLIENT is not None

	try:
		SEARCH_INDEX_CLIENT.get_index(SETTINGS.azure_search_index)
		return
	except ResourceNotFoundError:
		pass
	index = SearchIndex(
		name=SETTINGS.azure_search_index,
		fields=[
			SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
			SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
			SimpleField(name="file_name", type=SearchFieldDataType.String, filterable=True, sortable=True),
			SimpleField(name="title", type=SearchFieldDataType.String, filterable=True, sortable=True),
			SimpleField(name="blob_name", type=SearchFieldDataType.String, filterable=True, sortable=True),
			SimpleField(name="blob_url", type=SearchFieldDataType.String, filterable=False, sortable=False),
			SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
			SimpleField(name="chunk_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
			SearchableField(name="content", type=SearchFieldDataType.String),
			SimpleField(name="indexed_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
		],
	)
	SEARCH_INDEX_CLIENT.create_index(index)
def index_document_chunks(document: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
	require_services()
	assert SEARCH_CLIENT is not None
	SEARCH_CLIENT.upload_documents(documents=[*chunks, document])


def delete_existing_document(document_id: str) -> None:
	require_services()
	assert SEARCH_CLIENT is not None
	escaped_document_id = document_id.replace("'", "''")
	results = SEARCH_CLIENT.search(
		search_text="*",
		filter=f"document_id eq '{escaped_document_id}'",
		select=["chunk_id"],
		top=1000,
	)
	keys = [result["chunk_id"] for result in results]
	if keys:
		SEARCH_CLIENT.delete_documents(documents=[{"chunk_id": key} for key in keys])


def retrieve_chunks(question: str, top_k: int) -> list[dict[str, Any]]:
	require_services()
	assert SEARCH_CLIENT is not None
	# Try to request the preferred field names. If the index uses a different
	# schema (older deployments may have used `parent_id`/`chunk`), fall back
	# to an unselected search and map possible field names gracefully.
	# Try requesting preferred fields first; if the index doesn't contain those
	# fields the SDK may raise an HttpResponseError during iteration. In that
	# case fall back to an unselected search and map available fields.
	results = SEARCH_CLIENT.search(
		search_text=question,
		top=top_k,
		select=["chunk_id", "document_id", "content", "title", "page_number", "chunk_number"],
		include_total_count=False,
	)

	chunks: list[dict[str, Any]] = []
	try:
		for result in results:
			chunk_id = result.get("chunk_id") or result.get("id")
			document_id = result.get("document_id") or result.get("parent_id") or result.get("parentId")
			content = result.get("content") or result.get("chunk") or result.get("text")
			title = result.get("title")
			page_number = int(result.get("page_number") or result.get("page") or 0)
			chunk_number = int(result.get("chunk_number") or result.get("chunkNumber") or 0)
			score = result.get("@search.score")

			if not chunk_id:
				continue

			chunks.append(
				{
					"chunk_id": chunk_id,
					"page_number": page_number,
					"chunk_number": chunk_number,
					"document_id": document_id,
					"title": title,
					"content": content,
					"score": score,
				}
			)
	except HttpResponseError:
		# Fallback: retry without select and remap fields
		chunks = []
		results = SEARCH_CLIENT.search(
			search_text=question,
			top=top_k,
			include_total_count=False,
		)
		for result in results:
			chunk_id = result.get("chunk_id") or result.get("id")
			document_id = result.get("document_id") or result.get("parent_id") or result.get("parentId")
			content = result.get("content") or result.get("chunk") or result.get("text")
			title = result.get("title")
			page_number = int(result.get("page_number") or result.get("page") or 0)
			chunk_number = int(result.get("chunk_number") or result.get("chunkNumber") or 0)
			score = result.get("@search.score")

			if not chunk_id:
				continue

			chunks.append(
				{
					"chunk_id": chunk_id,
					"page_number": page_number,
					"chunk_number": chunk_number,
					"document_id": document_id,
					"title": title,
					"content": content,
					"score": score,
				}
			)
	return chunks


def answer_from_context(question: str, chunks: list[dict[str, Any]]) -> str:
	if not chunks:
		return "I could not find enough information in the indexed PDF to answer that question."

	context_lines = []
	for chunk in chunks:
		context_lines.append(f"[{chunk.get('title') or 'PDF'}] {chunk['content']}")
	context = "\n\n".join(context_lines)

	prompt = (
		"You answer questions using only the provided PDF context. "
		"Do not use outside knowledge or mention the web. "
		"If the answer is not in the context, say you could not find it in the document.\n\n"
		f"Context:\n{context}\n\n"
		f"Question: {question}"
	)

	# If the endpoint is an AI Projects agent endpoint, try calling the
	# Foundry Responses endpoint using AAD (REST). We attempt a few
	# api-version and request-shape combinations to maximize compatibility.
	project_error = None
	if "/api/projects" in (SETTINGS.azure_openai_endpoint or ""):
		try:
			if DefaultAzureCredential is None:
				raise HTTPException(status_code=503, detail="azure-identity not available for AAD authentication")

			def _call_projects_responses(base_url: str, agent_name: str, prompt_text: str) -> str:
				# Try DefaultAzureCredential first (interactive/dev auth).
				token = None
				try:
					cred = DefaultAzureCredential()
					token = cred.get_token("https://cognitiveservices.azure.com/.default").token
				except Exception:
					# Fallback: try Azure CLI to get an access token if installed
					try:
						import shutil, subprocess

						az = shutil.which("az") or r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
						if az and Path(az).exists():
							res = subprocess.run([az, "account", "get-access-token", "--resource", "https://cognitiveservices.azure.com/", "--query", "accessToken", "-o", "tsv"], capture_output=True, text=True, timeout=15)
							if res.returncode == 0:
								token = res.stdout.strip()
					except Exception:
						token = None

				if not token:
					raise Exception("AAD token acquisition failed via DefaultAzureCredential and Azure CLI")
				headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

				base = base_url.rstrip("/")
				url = f"{base}/agents/{agent_name}/endpoint/protocols/openai/responses"

				api_versions = [
					SETTINGS.azure_openai_api_version,
					"2025-08-07",
					"2025-05-01-preview",
					"2025-01-01-preview",
					"2024-12-01-preview",
					"2024-10-21",
					"2024-07-31-preview",
				]

				# Try several request body shapes commonly accepted by Responses APIs
				bodies = [
					{"input": prompt_text},
					{"messages": [{"role": "user", "content": prompt_text}]},
					{"input": {"messages": [{"role": "user", "content": prompt_text}]}}
				]

				for ver in [v for v in api_versions if v]:
					urlv = f"{url}?api-version={ver}"
					for body in bodies:
						try:
							resp = requests.post(urlv, headers=headers, json=body, timeout=30)
						except Exception:
							continue
						if resp.status_code >= 200 and resp.status_code < 300:
							try:
								data = resp.json()
							except Exception:
								return resp.text
							# Flexible parsing of common response shapes
							if isinstance(data, dict):
								if "output_text" in data:
									return data["output_text"]
								if "output" in data:
									o = data["output"]
									if isinstance(o, str):
										return o
									if isinstance(o, dict) and "content" in o:
										return o["content"]
								if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
									c = data["choices"][0]
									if isinstance(c, dict):
										if "message" in c and isinstance(c["message"], dict) and "content" in c["message"]:
											return c["message"]["content"]
										if "text" in c:
											return c["text"]
								if "value" in data and isinstance(data["value"], list) and data["value"]:
									# some Responses endpoints wrap outputs in value array
									v0 = data["value"][0]
									if isinstance(v0, dict) and "content" in v0:
										return v0["content"]
								# fallback to raw text
								return resp.text
						# non-2xx -> try next
				raise Exception("Projects Responses endpoint did not accept any tested api-version or payload shape")

			return _call_projects_responses(SETTINGS.azure_openai_endpoint, SETTINGS.azure_openai_deployment, prompt)
		except Exception as exc:
			project_error = exc

	if AZURE_OPENAI_CLIENT is None:
		raise HTTPException(status_code=503, detail="Azure OpenAI client is not configured")

	try:
		response = AZURE_OPENAI_CLIENT.chat.completions.create(
			model=SETTINGS.azure_openai_deployment,
			messages=[
				{"role": "system", "content": "You are a grounded assistant that answers only from the provided PDF context."},
				{"role": "user", "content": prompt},
			],
			temperature=0,
		)
	except Exception:
		# Primary deployment failed; attempt a safe fallback model once.
		try:
			response = AZURE_OPENAI_CLIENT.chat.completions.create(
				model="gpt-4o-mini",
				messages=[
					{"role": "system", "content": "You are a grounded assistant that answers only from the provided PDF context."},
					{"role": "user", "content": prompt},
				],
				temperature=0,
			)
		except Exception:
			# If both model calls fail (e.g., deployment not found), return a
			# simple context-only summary so the API remains usable.
			if chunks:
				summary_lines = [f"[{c.get('title') or 'PDF'}] {str(c.get('content') or '')}" for c in chunks[:3]]
				return "OpenAI unavailable; returning the top context snippets:\n\n" + "\n\n".join(summary_lines)
			return "OpenAI unavailable and no context could be retrieved to answer the question."
	message = response.choices[0].message.content or ""
	return message.strip()


@app.on_event("startup")
def startup() -> None:
	ensure_search_index()


@app.get("/health")
def health() -> dict[str, Any]:
	def _openai_status() -> str:
		if not (SETTINGS.azure_openai_endpoint and SETTINGS.azure_openai_api_key):
			return "not_configured"
		if AZURE_OPENAI_CLIENT is None:
			return "client_not_built"
		# Try a light-weight check to see if the configured deployment exists.
		try:
			AZURE_OPENAI_CLIENT.chat.completions.create(
				model=SETTINGS.azure_openai_deployment or "", messages=[{"role": "system", "content": "ping"}], temperature=0, max_tokens=1
			)
			return "deployment_ok"
		except Exception as err:
			msg = str(err)
			if "DeploymentNotFound" in msg or "deployment not found" in msg.lower():
				return "deployment_not_found"
			return "error"

	return {
		"status": "ok",
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"search_configured": bool(SETTINGS.azure_search_endpoint and SETTINGS.azure_search_key),
		"openai_configured": bool(SETTINGS.azure_openai_endpoint and SETTINGS.azure_openai_api_key),
		"openai_deployment": SETTINGS.azure_openai_deployment,
		"openai_status": _openai_status(),
	}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
	require_services()
	chunks = retrieve_chunks(request.question, request.top_k)
	try:
		answer = answer_from_context(request.question, chunks)
	except Exception as error:
		raise HTTPException(status_code=503, detail=f"Azure OpenAI request failed: {error}") from error
	sources = [
		SourceChunk(
			chunk_id=chunk["chunk_id"],
			page_number=int(chunk["page_number"]),
			score=chunk.get("score"),
			snippet=str(chunk["content"])[:400],
		)
		for chunk in chunks
	]
	return AskResponse(
		document_id=chunks[0].get("document_id") if chunks else "indexed-pdf",
		question=request.question,
		answer=answer,
		sources=sources,
		retrieved_at=datetime.now(timezone.utc),
	)


@app.get("/document")
def get_document() -> dict[str, Any]:
	require_services()
	assert SEARCH_CLIENT is not None
	# Try with preferred fields then fall back if the index uses different names.
	try:
		results = SEARCH_CLIENT.search(
			search_text="*",
			top=1,
			select=["chunk_id", "document_id", "title"],
		)
		for result in results:
			return {
				"document_id": result.get("document_id") or result["chunk_id"],
				"title": result.get("title") or "Indexed PDF",
			}
	except HttpResponseError:
		results = SEARCH_CLIENT.search(search_text="*", top=1)
		for result in results:
			doc_id = result.get("document_id") or result.get("parent_id") or result.get("chunk_id") or result.get("id")
			return {"document_id": doc_id, "title": result.get("title") or "Indexed PDF"}

	raise HTTPException(status_code=404, detail="Document not found")


@app.get("/", response_model=None)
def root() -> Any:
	if INDEX_FILE.exists():
		return FileResponse(INDEX_FILE)
	return {"message": "PDF RAG backend is running"}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
