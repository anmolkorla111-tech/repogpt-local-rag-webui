\# RepoGPT – Local RAG + Ollama Web UI (CPU-friendly)



RepoGPT is a \*\*local, CPU-friendly Retrieval-Augmented Generation (RAG)\*\* assistant that answers questions about any code repository using:



\- \*\*Ollama\*\* (local LLM)

\- \*\*Embeddings\*\*: `nomic-embed-text`

\- \*\*Vector Search\*\*: FAISS

\- \*\*Backend\*\*: FastAPI

\- \*\*UI\*\*: Simple Web Chat + Sources list



✅ Runs locally on low-spec machines (CPU + 8GB RAM)  

✅ Shows answers with \*\*source file names\*\*  

✅ Works offline after model download



---



\## Demo (Screenshots)



> Save images inside `/screenshots` and keep these names.



\- Home UI  

&nbsp; !\[Home](screenshots/01-home.png)



\- Answer + Sources  

&nbsp; !\[Answer](screenshots/02-answer.png)



\- Sources panel  

&nbsp; !\[Sources](screenshots/03-sources.png)



---



\## Architecture



\*\*Browser UI\*\* → \*\*FastAPI\*\* (`/api/ask`) → \*\*FAISS Retrieval\*\* → \*\*Ollama LLM\*\* → Answer + Sources



---



\## Features



\- Ask repo questions like: `What does main.py do?`

\- Shows answer + \*\*sources\*\*

\- Health check: `/api/health`

\- Web UI: `/`



---



\## Prerequisites



\- Windows 10/11

\- Python 3.10+

\- Ollama installed \& running



---



\## Setup



\### 1) Pull models

```powershell

ollama pull nomic-embed-text

ollama pull qwen2.5:3b

