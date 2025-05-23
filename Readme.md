# AITeller

AITeller is an intelligent banking assistant that uses large language models and various AI techniques to provide banking information and assistance via voice and text interfaces.

## Overview

This project integrates several AI technologies:
- LLM-powered agent with banking knowledge
- RAG (Retrieval-Augmented Generation) for enhanced responses
- Speech-to-Text with Whisper
- Text-to-Speech with Google Cloud TTS
- VoIP connectivity with Twilio

## Components

### Conversation Management

The `ConversationManager` class maintains conversation history and provides context-aware responses using few-shot prompting techniques.

### AI Agent

The system uses LangChain's agent framework with the following capabilities:
- Tool usage for structured data lookup
- RAG for augmenting responses with relevant knowledge
- Access to banking account information

### Tools

- Bank Account Lookup: Retrieves account details by name or account number
- RAG Retriever: Provides information from embedded knowledge documents

### Voice Processing

- Speech recognition via Whisper
- Text-to-speech synthesis via Google Cloud TTS
- VoIP call handling with Twilio

## Data Sources

- Dummy bank account data (JSON)
- RAG documents for specialized knowledge

## Vector Storage

The system uses FAISS for efficient vector storage and retrieval:
- Document embedding with Ollama embeddings
- Vector similarity search for RAG

## API Server

- FastAPI/Flask backend
- Uvicorn ASGI server

## Setup and Installation

1. Install dependencies:
   ```
   pip install -r Backend/requirements.txt
   ```

2. Set up environment variables (for Google Cloud TTS, if used)

3. Embed documents for RAG:
   ```
   python -m Backend.utils.embed_dummy_rag_doc
   ```

4. Run the server:
   ```
   python -m Backend
   ```

## Usage

The system can be interacted with via:
- API endpoints
- VoIP calls (when configured with Twilio)

Example query:
```
What is the balance for Alice Johnson?
```

## Architecture

```
Backend/
├── __init__.py
├── __main__.py - Entry point
├── app/ - Web application
│   └── routes/
├── config/ - Configuration
│   ├── app_factory.py - App initialization
│   ├── app_logger.py - Logging setup
│   ├── register_routes.py - Route configuration
│   └── uvicorn_config.py - ASGI server configuration
├── services/ - Core functionality
│   ├── google_tts.py - Text-to-speech
│   ├── llm.py - LLM agent integration
│   ├── twilio_voip.py - Voice call handling
│   └── whisper_stt.py - Speech recognition
├── tools/ - Agent tools
│   └── dummy.py - Bank data lookup
└── utils/ - Helpers
    ├── dummy_bank_data.json - Sample banking data
    ├── dummy_rag_doc.txt - RAG test document
    ├── embed_documents.py - Document embedding
    └── embed_dummy_rag_doc.py - RAG setup
```

## Development

The project is structured to easily extend with additional:
- Banking tools
- Knowledge documents
- Interface modalities
