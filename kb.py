import os, json, math
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
EMBED_MODEL = "gemini-embedding-001"
INDEX_FILE = "kb_index.json"

# ShopEasy help content, already split into chunks (one idea per chunk)
DOCS = [
    {"id": "shipping-1", "topic": "shipping", "text": "Standard delivery takes 3 to 5 working days in metro cities and 5 to 8 working days elsewhere. Delivery is free on orders above ₹499; below that, a ₹49 delivery fee applies."},
    {"id": "shipping-2", "topic": "shipping", "text": "Express delivery is available in Pune, Mumbai, Bengaluru and Delhi for ₹99 and arrives the next working day if ordered before 2 pm."},
    {"id": "returns-1", "topic": "returns", "text": "Most items can be returned within 7 days of delivery if unused and in original packaging. Refunds go back to the original payment method within 5 to 7 working days after the return is picked up."},
    {"id": "returns-2", "topic": "returns", "text": "Innerwear, personal care products and gift cards cannot be returned for hygiene and security reasons."},
    {"id": "warranty-1", "topic": "warranty", "text": "Electronics such as earbuds, smartwatches and phones carry a 1-year manufacturer warranty. Warranty claims are handled by the brand's service centre; ShopEasy can create a ticket to help arrange pickup."},
    {"id": "warranty-2", "topic": "warranty", "text": "Physical damage such as a cracked screen or water damage is not covered by the standard warranty."},
    {"id": "payments-1", "topic": "payments", "text": "ShopEasy accepts UPI, debit and credit cards, net banking and cash on delivery. Cash on delivery is available for orders up to ₹10,000."},
    {"id": "payments-2", "topic": "payments", "text": "If money was debited but the order failed, the amount is automatically reversed to the source account within 3 working days."},
    {"id": "account-1", "topic": "account", "text": "To change the delivery address, open My Orders and choose Edit Address. The address can only be changed before the order is shipped."},
]


def embed(texts: list[str], task: str) -> list[list[float]]:
    """Turn texts into vectors with Gemini's embedding model."""
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return [e.values for e in resp.embeddings]


def cosine(a: list[float], b: list[float]) -> float:
    """Similarity of two vectors: 1.0 = same meaning, near 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def load_index() -> list[dict]:
    """Embed the docs once and cache them in a JSON file."""
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, encoding="utf-8") as f:
            return json.load(f)
    vectors = embed([d["text"] for d in DOCS], "RETRIEVAL_DOCUMENT")
    index = [{**d, "vector": v} for d, v in zip(DOCS, vectors)]
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return index


INDEX = load_index()


def search_help(query: str, k: int = 3) -> dict:
    """Search ShopEasy's help articles (shipping, returns, warranty, payments, account).
    Use this for questions about store policies and how things work.
    Returns the most relevant passages with a similarity score (higher = more relevant)."""
    try:
        qv = embed([query], "RETRIEVAL_QUERY")[0]
    except errors.APIError as e:
        return {"status": "error", "message": f"Search unavailable (code {e.code})."}
    scored = sorted(((cosine(qv, d["vector"]), d) for d in INDEX), key=lambda p: p[0], reverse=True)
    hits = [{"text": d["text"], "topic": d["topic"], "score": round(s, 3)} for s, d in scored[:k]]
    return {"status": "success", "results": hits}


if __name__ == "__main__":
    questions = [
        "How long does delivery take?",
        "Is my cracked smartwatch screen covered?",
        "Can I pay cash?",
        "Do you sell cricket bats?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        for h in search_help(q)["results"]:
            print(f"  [{h['score']}] ({h['topic']}) {h['text'][:70]}...")