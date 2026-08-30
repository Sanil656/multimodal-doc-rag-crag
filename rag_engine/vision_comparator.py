import os
import base64
import io
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .chain import get_llm, format_docs_with_metadata


VISUAL_ANALYSIS_PROMPT = """You are an expert technical visual analyst and chart pattern specialist.
Analyze this uploaded image (such as a stock market candlestick chart, technical diagram, or visual pattern).

Your task:
1. Describe the key visual structures in detail:
   - Specific Candlestick Formations (e.g., Hammer, Shooting Star, Bullish/Bearish Engulfing, Doji, Morning/Evening Star, Head & Shoulders, Double Top/Bottom, Flag, Triangle, etc.)
   - Price Trend context leading into the pattern (Uptrend, Downtrend, Consolidation/Sideways)
   - Candle characteristics: Real body size, upper/lower wick lengths, open/close relative positions, color (green/bullish, red/bearish)
   - Notable indicators, support/resistance levels, or volume spikes if visible in the chart.
2. Provide a concise summary of the primary pattern(s) identified and 3-5 search keywords to look up in the reference technical book.

Format your response clearly:
**Primary Patterns Identified**: <List of patterns>
**Trend Context**: <Uptrend / Downtrend / Sideways>
**Visual Characteristics**: <Detailed candle and structure details>
**Key Search Queries**: <Keyword 1, Keyword 2, Keyword 3>
"""

COMPARISON_PROMPT = """You are an expert technical analyst comparing a REAL-WORLD UPLOADED CHART/IMAGE against a REFERENCE GUIDEBOOK/MANUAL.

Below is the visual analysis of the user's uploaded chart, along with relevant excerpts retrieved from the reference book/guide.

--- VISUAL ANALYSIS OF THE UPLOADED IMAGE ---
{visual_analysis}

--- EXCERPTS FROM THE REFERENCE BOOK (WITH PAGE CITATIONS) ---
{book_context}

--- ADDITIONAL USER INSTRUCTIONS / FOCUS ---
{user_instructions}

YOUR TASK:
Perform a comprehensive, rigorous comparison between the uploaded actual image and the reference book's rules.

Structure your response with the following sections:

### 1. 🎯 Pattern Identification & Book Match
- Clearly name the pattern found in the image.
- State the matching Chapter / Page Number(s) from the reference book (e.g., `[Source: Book | Page 42]`).

### 2. 📋 Criteria Verification Checklist (Theory vs. Actual)
Compare the book's mandatory criteria with what is visible in the actual image:
- [ ] **Condition 1 (from Book Page X)**: [Met / Partially Met / Not Met] - *Explanation with comparison*
- [ ] **Condition 2 (from Book Page X)**: [Met / Partially Met / Not Met] - *Explanation with comparison*
- [ ] **Condition 3 (Trend / Confirmation)**: [Met / Partially Met / Not Met] - *Explanation with comparison*

### 3. 📈 Book's Predicted Outcome & Action Plan
- What does the reference book predict should happen next? (e.g., Bullish Reversal, Bearish Continuation, Breakout target).
- What are the book's recommended Entry, Stop Loss, and Profit Target guidelines?

### 4. ⚠️ Discrepancies & Risk Warnings
- Any false signals, missing confirmation (e.g., volume, next candle confirmation), or deviations from the book's ideal pattern.

### 5. 💡 Overall Verdict & Confidence Score
- **Pattern Match Confidence**: [e.g., 85% Match]
- **Summary Conclusion**: Concise takeaway.
"""


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_media_type(filename: str) -> str:
    """Determine MIME type from filename."""
    ext = filename.lower().split(".")[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "webp":
        return "image/webp"
    elif ext == "bmp":
        return "image/bmp"
    return "image/jpeg"


def analyze_and_compare_image_with_book(
    image_bytes: bytes,
    image_filename: str,
    vector_store,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    user_instructions: str = "",
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Complete end-to-end multimodal pipeline:
    1. Analyzes chart/image features using Vision LLM.
    2. Searches the reference book vector store for matching patterns and rules.
    3. Generates a comparative verification report with page citations.
    """
    media_type = get_image_media_type(image_filename)
    b64_image = encode_image_to_base64(image_bytes)

    # Initialize Vision Model
    if provider == "gemini":
        vision_model_name = model_name or "gemini-1.5-flash"
        from langchain_google_genai import ChatGoogleGenerativeAI
        vision_llm = ChatGoogleGenerativeAI(
            model=vision_model_name,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )
        
        # Step 1: Multimodal Visual Inspection
        image_url_payload = f"data:{media_type};base64,{b64_image}"
        vision_message = HumanMessage(
            content=[
                {"type": "text", "text": VISUAL_ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": image_url_payload},
            ]
        )
        visual_analysis_res = vision_llm.invoke([vision_message])
        visual_analysis_text = visual_analysis_res.content

    elif provider == "ollama":
        # For Ollama Vision models (e.g. llama3.2-vision, llava) or fallback
        from langchain_ollama import ChatOllama
        # Use vision-capable model if specified, or default to llama3.2-vision / llava
        ollama_vision_model = model_name if ("vision" in model_name.lower() or "llava" in model_name.lower()) else "llama3.2-vision"
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        vision_llm = ChatOllama(
            model=ollama_vision_model,
            base_url=base_url,
            temperature=0.2,
        )
        
        vision_message = HumanMessage(
            content=[
                {"type": "text", "text": VISUAL_ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": f"data:{media_type};base64,{b64_image}"},
            ]
        )
        try:
            visual_analysis_res = vision_llm.invoke([vision_message])
            visual_analysis_text = visual_analysis_res.content
        except Exception as e:
            # Fallback if standard model doesn't support images directly
            visual_analysis_text = f"Visual inspection performed. Error with Ollama vision model: {str(e)}. Please ensure 'llama3.2-vision' or 'llava' is pulled in Ollama."

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Step 2: Retrieve matching theoretical pages from reference book vector store
    # Query vector store using the identified pattern keywords from visual analysis
    search_query = f"{visual_analysis_text}\n{user_instructions}"
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(search_query)

    # Step 3: Synthesize side-by-side comparison
    formatted_book_context = format_docs_with_metadata(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(COMPARISON_PROMPT)
    llm = get_llm(
        provider=provider,
        model_name=model_name,
        temperature=0.2,
        api_key=api_key,
        base_url=base_url,
    )

    comparator_chain = prompt | llm | StrOutputParser()
    comparison_report = comparator_chain.invoke({
        "visual_analysis": visual_analysis_text,
        "book_context": formatted_book_context,
        "user_instructions": user_instructions if user_instructions else "Compare all identified patterns against the book's rules.",
    })

    return {
        "visual_analysis": visual_analysis_text,
        "comparison_report": comparison_report,
        "source_documents": retrieved_docs,
    }
