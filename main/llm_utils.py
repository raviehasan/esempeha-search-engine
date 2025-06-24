import os
import requests
import hashlib
from django.conf import settings
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)

def get_cache_key(query, documents):
    """Generate cache key for query and documents"""
    # Create a hash of the query and document IDs
    doc_ids = [doc.get('doc_id', doc.get('id', '')) for doc in documents]
    content = f"{query}:{':'.join(sorted(doc_ids))}"
    return f"llm_summary:{hashlib.md5(content.encode()).hexdigest()}"

def get_llm_summary(query: str, documents: list, max_doc_length=700):
    """
    Generates a summary using HuggingFace Inference API with caching.
    """
    api_key = settings.HUGGINGFACE_API_KEY
    model_id = settings.LLM_MODEL_ID
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    if not api_key:
        logger.warning("HUGGINGFACE_API_KEY not found. LLM summarization disabled.")
        return "LLM summarization is unavailable (API key missing)."

    if not documents:
        return "No documents provided."

    # Check cache first
    cache_key = get_cache_key(query, documents)
    cached_summary = cache.get(cache_key)
    if cached_summary:
        logger.info(f"Using cached LLM summary for query: {query}")
        return cached_summary

    headers = {"Authorization": f"Bearer {api_key}"}
    
    context_parts = []
    for i, doc in enumerate(documents[:3]):
        doc_text = doc.get('text', '')
        doc_title = doc.get('title', 'Document')
        snippet_text = str(doc_text) if doc_text else "No abstract available."
        snippet = snippet_text[:max_doc_length] + "..." if len(snippet_text) > max_doc_length else snippet_text
        context_parts.append(f"Document {i+1} (Title: {doc_title}):\n{snippet}")
    
    context_str = "\n\n".join(context_parts)

    prompt = (
        f"User Query: \"{query}\"\n\n"
        f"Using only the information provided in the excerpts below, generate a concise, factual answer to the user's query. "
        f"Synthesize information across the documents where possible. If no clear answer can be derived, summarize the relevant insights. "
        f"Do not repeat each document individually. Focus on providing a unified answer.\n\n"
        f"Document Excerpts:\n{context_str}\n\n"
        f"Answer:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.5,
            "return_full_text": False,
            "wait_for_model": True,
        },
        "options": {
            "use_cache": False
        }
    }

    try:
        logger.info(f"Sending request to LLM: {model_id} with query: {query}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=45) 
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                summary = result[0]["generated_text"].strip()
                
                # Cache the summary
                cache.set(cache_key, summary, timeout=getattr(settings, 'LLM_CACHE_TIMEOUT', 3600))
                
                logger.info(f"LLM summary received for query '{query}': {summary[:100]}...")
                return summary
            else:
                logger.error(f"Unexpected LLM API response format for query '{query}': {result}")
                return "Could not generate summary due to API response format."
        else:
            error_content = response.text
            logger.error(f"LLM API request failed for query '{query}' with status {response.status_code}: {error_content}")
            
            if response.status_code == 401:
                return "LLM API request failed: Unauthorized (check API key)."
            elif response.status_code == 429:
                return "LLM service is currently busy (rate limit exceeded). Please try again later."
            elif response.status_code >= 500:
                return f"LLM service unavailable (server error {response.status_code}). Please try again later."
            return f"Failed to get summary from LLM (HTTP {response.status_code})."

    except requests.exceptions.Timeout:
        logger.error(f"LLM API request timed out for query '{query}'.")
        return "LLM request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API request failed for query '{query}': {e}")
        return "Failed to get summary from LLM due to a connection or API error."
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting LLM summary for query '{query}': {e}", exc_info=True)
        return "An unexpected error occurred while generating the summary."

