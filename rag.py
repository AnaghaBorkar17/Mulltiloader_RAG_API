import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vectorstore import search_documents_with_score, create_vector_store

# Local free model: runs on CPU without any OpenAI API key or credits
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

_model = None
_tokenizer = None
_local_model_failed = False
_local_model_error = None

# Similarity threshold: cosine similarity threshold for sentence-transformers/all-MiniLM-L6-v2
# Rejects completely unrelated queries (e.g. asking recipe ingredients against a sales report)
SIMILARITY_THRESHOLD = 0.25

FALLBACK_MESSAGE = "I could not find this information in the uploaded documents."


def get_llm():
    """
    Lazy-load the local LLM and tokenizer in memory as a singleton.
    Loads once on startup / first query and stays in memory for fast inference.
    Catches memory limits gracefully on cloud hosts (e.g. Streamlit Cloud).
    """
    global _model, _tokenizer, _local_model_failed, _local_model_error
    if _local_model_failed:
        return None, None

    if _model is None or _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            _local_model_failed = True
            _local_model_error = str(e)
            print(f"Warning: Could not load local LLM ({e}). Falling back to extractive mode or OpenAI.")
            return None, None

    return _model, _tokenizer


def generate_openai_answer(context: str, question: str, api_key: str) -> str:
    """
    Generate an answer using OpenAI's fast, low-memory gpt-4o-mini API.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        system_prompt = (
            "You are a helpful and precise assistant. "
            "Answer the user's question directly and concisely (in 1 to 2 sentences) using ONLY the facts provided in the Context. "
            f"If the context does not contain enough information to answer the question, respond strictly with: '{FALLBACK_MESSAGE}'"
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI query error: {str(e)}"


def generate_local_answer(context: str, question: str) -> str:
    """
    Generate a question-specific, concise answer using the local Hugging Face LLM.
    If local model cannot load (e.g. low memory cloud environment), fall back to extractive summary.
    """
    model, tokenizer = get_llm()

    if model is None or tokenizer is None:
        # Fallback to direct context extraction (resilient on Streamlit Cloud 1GB RAM)
        first_section = context.split("\n---\n")[0].strip()
        lines = [line.strip() for line in first_section.split("\n") if line.strip()]
        preview = " ".join(lines[:4])
        return f"{preview}"

    system_prompt = (
        "You are a helpful and precise assistant. "
        "Answer the user's question directly and concisely (in 1 to 2 sentences) using ONLY the facts provided in the Context. "
        "If the context does not contain enough information to answer the question, respond strictly with: "
        f"'{FALLBACK_MESSAGE}'"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated response tokens
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer


def answer_question(
    question: str,
    k: int = 4,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    openai_api_key: str = None
):
    """
    Complete RAG pipeline:
    1. Semantic similarity retrieval with cosine scoring
    2. Filter out irrelevant chunks below similarity threshold
    3. Generate question-specific answer with local LLM or OpenAI
    4. Return concise answer + deduplicated sources
    """
    clean_question = question.strip()
    if not clean_question:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": []
        }

    # 1. Retrieve top-k chunks with similarity scores
    scored_results = search_documents_with_score(clean_question, k=k)

    if not scored_results:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": []
        }

    # 2. Filter by similarity threshold to eliminate cross-document contamination
    relevant_chunks = [
        doc for doc, score in scored_results
        if score >= similarity_threshold
    ]

    # If no chunk meets relevance threshold, reject immediately without calling LLM
    if not relevant_chunks:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": []
        }

    # 3. Build context from relevant chunks
    context = "\n---\n".join(doc.page_content for doc in relevant_chunks)

    # 4. Generate answer (OpenAI if key provided, else local LLM / extractive)
    effective_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if effective_api_key and effective_api_key.startswith("sk-"):
        raw_answer = generate_openai_answer(context, clean_question, effective_api_key)
    else:
        raw_answer = generate_local_answer(context, clean_question)

    # If the LLM indicates information is not in the context
    if FALLBACK_MESSAGE.lower() in raw_answer.lower():
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": []
        }

    # 5. Extract deduplicated sources that contributed to the answer
    seen = set()
    sources = []
    for doc in relevant_chunks:
        src = doc.metadata.get("source", "Unknown")
        ft = doc.metadata.get("file_type", "unknown")
        if (src, ft) not in seen:
            seen.add((src, ft))
            sources.append({
                "source": src,
                "file_type": ft
            })

    return {
        "answer": raw_answer,
        "sources": sources
    }
