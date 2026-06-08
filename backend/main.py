"""
main.py — FastAPI backend for Mom Test Coach
RAG retrieval from Supabase + LLM via Groq
Features: PDF/PPT text extraction, configurable question count,
          Excel export of questions, Excel upload + AI scoring
"""

import os
import io
import re
import json
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(dotenv_path=Path(__file__).parent / ".env")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 6

app = FastAPI(title="Mom Test Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.environ["HF_API_KEY"]
HF_EMBEDDING_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def get_embedding(text: str) -> list[float]:
    resp = httpx.post(
        HF_EMBEDDING_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text},
        timeout=15,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding error: {resp.text}")
    result = resp.json()
    if isinstance(result[0], list):
        import statistics
        return [statistics.mean(col) for col in zip(*result[0])]
    return result

print("BuildRight backend ready.")


# ── Models ────────────────────────────────────────────────────────────────
class GenerateQuestionsRequest(BaseModel):
    segment: str
    product_idea: str
    target_customer: str
    problem_hypothesis: str
    additional_context: str = ""
    num_questions: int = 15

class Question(BaseModel):
    q: str
    why: str
    category: str = ""

class GenerateQuestionsResponse(BaseModel):
    questions: list[Question]
    book_excerpts_used: list[str]


# ── File text extraction ────────────────────────────────────────────────────
def extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()

def extract_pptx_text(data: bytes) -> str:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        lines = [f"[Slide {i}]"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs)
                    if line.strip():
                        lines.append(line)
        parts.append("\n".join(lines))
    return "\n\n".join(parts).strip()


# ── RAG ───────────────────────────────────────────────────────────────────────
def retrieve_chunks(query: str, top_k: int = TOP_K) -> list[str]:
    query_embedding = embedder.encode(query).tolist()
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"query_embedding": query_embedding, "match_count": top_k}
    resp = httpx.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_mom_test_chunks",
        headers=headers, json=payload, timeout=15,
    )
    if resp.status_code != 200:
        print(f"Supabase error: {resp.text}")
        return []
    return [r["content"] for r in resp.json()]


# ── Groq ──────────────────────────────────────────────────────────────────────
def call_groq(system: str, user: str, max_tokens: int = 2500) -> str:
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        timeout=45,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groq error: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]

def parse_json_array(raw: str):
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except Exception:
        m = re.search(r"\[.*\]", clean, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise HTTPException(status_code=500, detail="Failed to parse LLM response")

def parse_json_object(raw: str):
    clean = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(clean)
    except Exception:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise HTTPException(status_code=500, detail="Failed to parse LLM response")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.post("/extract-file")
async def extract_file(file: UploadFile = File(...)):
    data = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".pdf"):
            text = extract_pdf_text(data)
        elif name.endswith(".pptx"):
            text = extract_pptx_text(data)
        elif name.endswith((".txt", ".md")):
            text = data.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, PPTX, TXT or MD.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read file: {e}")
    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="No text found. This may be a scanned/image-only file.")
    return {"filename": file.filename, "char_count": len(text), "text": text[:12000]}


@app.post("/generate-questions", response_model=GenerateQuestionsResponse)
def generate_questions(req: GenerateQuestionsRequest):
    n = max(5, min(25, req.num_questions))
    rag_query = f"customer discovery questions {req.problem_hypothesis} {req.target_customer} past behavior workarounds commitment"
    book_chunks = retrieve_chunks(rag_query, top_k=TOP_K)
    book_context = "\n\n---\n\n".join(book_chunks)

    system_prompt = f"""You are a world-class customer discovery coach with deep knowledge of The Mom Test by Rob Fitzpatrick.

Relevant passages from The Mom Test book to ground your questions:

<book_excerpts>
{book_context}
</book_excerpts>

THE MOM TEST RULES — apply every one:
1. NEVER mention the product idea or feature being validated
2. NEVER ask "would you use this?" or "do you like this?" — compliment-bait
3. NEVER ask about hypothetical future behavior — only past/present
4. ALWAYS ask about the customer's life, struggles, past actions
5. Focus on frequency, severity, and current workarounds
6. Dig for specific episodes: "tell me about the last time..." "walk me through..."
7. For B2C: language must be simple, conversational, non-technical

Generate exactly {n} questions, organised across these research stages:
- Context & background (their world, role, daily life)
- Problem discovery (the pain, when it shows up)
- Current behavior (what they do TODAY, workarounds, tools)
- Severity & cost (how much the problem costs them in time/money/stress)
- Past attempts (what they've already tried, searched for, paid for)

Respond ONLY with a valid JSON array. No preamble, no markdown fences.
Format:
[
  {{"q": "question text", "why": "which Mom Test rule + what signal you're hunting", "category": "one of: Context | Problem | Current Behavior | Severity | Past Attempts"}}
]"""

    user_msg = f"""Generate {n} Mom Test-compliant discovery questions.

Customer type: {req.segment.upper()}
Target customer: {req.target_customer}
Problem hypothesis: {req.problem_hypothesis}

UPLOADED DOCUMENT (the product context — read it carefully):
{req.additional_context[:6000] or 'None provided'}

YOUR JOB:
1. Read the uploaded document and product idea. Break the product down into the
   core ASSUMPTIONS it depends on to succeed with the customer. (e.g. "customer
   will lock money in a deposit", "rejection is a real felt pain", "fuel rewards
   motivate them", "they trust the bank after being rejected").
2. Detect the SITUATION from the context — standalone new product, a feature on
   something they already use, or a cross-sell inside an existing journey — and
   frame questions to fit it. Do not assume; infer from the document.
3. Write questions that TEST each assumption through the customer's REAL PAST
   BEHAVIOR. Past actions only — never "would you" hypotheticals.
4. You MAY ask about real things in their life that relate to the problem:
   their history with fixed deposits, locking money away, credit card
   applications and rejections, how they pay for fuel, how they handle tight
   months. These are the customer's real history — fair game and necessary.
5. You must NEVER name, describe, or pitch THIS specific product or its features
   (no "FD-backed card", no "FIRST Power", no reward structure). Test the
   assumption through behavior, not the solution.
6. Spread the {n} questions across the assumptions so each key bet gets probed.

Do NOT reveal or hint at the product idea: "{req.product_idea}"
Ground every question in the book excerpts AND the uploaded document."""
    raw = call_groq(system_prompt, user_msg)
    data = parse_json_array(raw)
    questions = [Question(q=i["q"], why=i.get("why", ""), category=i.get("category", "")) for i in data]
    return GenerateQuestionsResponse(questions=questions, book_excerpts_used=book_chunks[:2])


@app.post("/export-excel")
def export_excel(payload: dict):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    questions = payload.get("questions", [])
    num_customers = max(1, min(50, payload.get("num_customers", 10)))
    product_idea = payload.get("product_idea", "")
    target = payload.get("target_customer", "")

    wb = Workbook()
    ws = wb.active
    ws.title = "Interview Capture"

    ws["A1"] = "Mom Test — Customer Interview Capture"
    ws["A1"].font = Font(bold=True, size=14, name="Arial")
    ws["A2"] = f"Target customer: {target}"
    ws["A2"].font = Font(italic=True, size=10, name="Arial", color="666666")
    ws["A3"] = "Fill each customer's answer. Adjust 'Weightage' (higher = more important). Then re-upload."
    ws["A3"].font = Font(italic=True, size=9, name="Arial", color="999999")

    header_row = 5
    headers = ["#", "Category", "Question", "Weightage"] + [f"Customer {i+1}" for i in range(num_customers)]
    for col, h in enumerate(headers, start=1):
        c = ws.cell(row=header_row, column=col, value=h)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        c.fill = PatternFill("solid", start_color="1A1916")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    thin = Side(style="thin", color="DDDDDD")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for idx, q in enumerate(questions):
        r = header_row + 1 + idx
        ws.cell(row=r, column=1, value=idx + 1).font = Font(name="Arial", size=10)
        ws.cell(row=r, column=2, value=q.get("category", "")).font = Font(name="Arial", size=9, color="666666")
        qc = ws.cell(row=r, column=3, value=q.get("q", ""))
        qc.font = Font(name="Arial", size=10)
        qc.alignment = Alignment(wrap_text=True, vertical="top")
        wc = ws.cell(row=r, column=4, value=3)
        wc.font = Font(name="Arial", size=10, color="0000FF")
        wc.alignment = Alignment(horizontal="center")
        wc.fill = PatternFill("solid", start_color="FFFDE7")
        for cust in range(num_customers):
            cell = ws.cell(row=r, column=5 + cust, value="")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = border

    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 11
    for cust in range(num_customers):
        ws.column_dimensions[ws.cell(row=header_row, column=5 + cust).column_letter].width = 40

    meta = wb.create_sheet("_meta")
    meta["A1"] = "product_idea"; meta["B1"] = product_idea
    meta["A2"] = "target_customer"; meta["B2"] = target
    meta["A3"] = "segment"; meta["B3"] = payload.get("segment", "b2c")
    meta["A4"] = "problem_hypothesis"; meta["B4"] = payload.get("problem_hypothesis", "")
    meta.sheet_state = "hidden"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=mom_test_interviews.xlsx"},
    )


@app.post("/analyze-excel")
async def analyze_excel(file: UploadFile = File(...)):
    from openpyxl import load_workbook

    data = await file.read()
    wb = load_workbook(io.BytesIO(data), data_only=True)
    ws = wb["Interview Capture"] if "Interview Capture" in wb.sheetnames else wb.active

    meta_ctx = {"product_idea": "", "target_customer": "", "segment": "b2c", "problem_hypothesis": ""}
    if "_meta" in wb.sheetnames:
        m = wb["_meta"]
        for row in m.iter_rows(min_row=1, max_row=10, values_only=True):
            if row and row[0] in meta_ctx:
                meta_ctx[row[0]] = row[1] or ""

    header_row = None
    for r in range(1, 12):
        vals = [str(ws.cell(row=r, column=c).value) for c in range(1, 6)]
        if "Question" in vals:
            header_row = r
            break
    if header_row is None:
        raise HTTPException(status_code=422, detail="Could not find the question table. Use the exported template.")

    customer_cols = []
    col = 5
    while True:
        h = ws.cell(row=header_row, column=col).value
        if not h:
            break
        customer_cols.append((col, str(h)))
        col += 1

    rows = []
    r = header_row + 1
    while True:
        q = ws.cell(row=r, column=3).value
        if not q:
            break
        weight = ws.cell(row=r, column=4).value or 3
        answers = {}
        for ccol, cname in customer_cols:
            ans = ws.cell(row=r, column=ccol).value
            if ans and str(ans).strip():
                answers[cname] = str(ans).strip()
        rows.append({"question": str(q), "weight": float(weight), "answers": answers})
        r += 1

    customers = {}
    for ccol, cname in customer_cols:
        qa = [{"q": row["question"], "weight": row["weight"], "a": row["answers"][cname]}
              for row in rows if cname in row["answers"]]
        if qa:
            customers[cname] = qa

    if not customers:
        raise HTTPException(status_code=422, detail="No customer answers found in the file.")

    book_chunks = retrieve_chunks("validation signals commitment workarounds real pain evidence scoring", top_k=TOP_K)
    book_context = "\n\n---\n\n".join(book_chunks)

    transcript = ""
    for cname, qa in customers.items():
        transcript += f"\n=== {cname} ===\n"
        for item in qa:
            transcript += f"[weight {item['weight']}] Q: {item['q']}\nA: {item['a']}\n"

    system_prompt = f"""You are a ruthlessly objective customer validation analyst trained on The Mom Test by Rob Fitzpatrick.

Relevant book passages to ground your analysis:
<book_excerpts>
{book_context}
</book_excerpts>

You will receive interview answers from MULTIPLE customers, each question carrying a weightage.
For EACH customer, judge evidence quality (real specific pain, frequency, current workarounds, willingness to act). Weight higher-weightage questions more.
Then produce an AGGREGATE view across all customers.

SCORING (0-100):
- 75-100 GO: most customers show real, frequent, costly pain with active workarounds
- 50-74 PIVOT: mixed; some signal but key assumptions unproven
- 0-49 NO-GO: mostly polite noise, no real pain, no workarounds

Respond ONLY with valid JSON, no markdown fences:
{{
  "aggregate_score": <0-100>,
  "verdict": "<GO|PIVOT|NO-GO>",
  "verdict_reason": "<2 sentences>",
  "per_customer": [
    {{"name": "Customer 1", "score": <0-100>, "signal": "<strong|medium|weak>", "summary": "<1 sentence>"}}
  ],
  "strong_signals": ["<pattern across customers>"],
  "weak_signals": ["<concern or gap>"],
  "mom_test_violations": ["<answers that look like politeness, not evidence>"],
  "next_questions": ["<sharper follow-up for next round>"],
  "recommendation": "<3-4 sentences: what to do next>"
}}"""

    user_msg = f"""Product idea (NOT shown to customers): {meta_ctx['product_idea']}
Target: {meta_ctx['target_customer']} ({meta_ctx['segment'].upper()})
Problem hypothesis: {meta_ctx['problem_hypothesis']}

Interview transcript across {len(customers)} customers:
{transcript[:9000]}

Score each customer and give the aggregate. Ground analysis in the book excerpts."""

    raw = call_groq(system_prompt, user_msg, max_tokens=2500)
    report = parse_json_object(raw)
    report["num_customers"] = len(customers)
    report["book_excerpts_used"] = book_chunks[:2]
    return report


class ManualAnswer(BaseModel):
    q: str
    a: str

class AnalyzeManualRequest(BaseModel):
    segment: str
    product_idea: str
    target_customer: str
    problem_hypothesis: str
    answers: list[ManualAnswer]


@app.post("/analyze-manual")
def analyze_manual(req: AnalyzeManualRequest):
    """Score a single customer's pasted answers (no Excel needed)."""
    answered = [a for a in req.answers if a.a.strip()]
    if len(answered) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 answers.")

    book_chunks = retrieve_chunks("validation signals commitment workarounds real pain evidence scoring", top_k=TOP_K)
    book_context = "\n\n---\n\n".join(book_chunks)

    transcript = "\n".join(f"Q: {a.q}\nA: {a.a}" for a in answered)

    system_prompt = f"""You are a ruthlessly objective customer validation analyst trained on The Mom Test by Rob Fitzpatrick.

Relevant book passages:
<book_excerpts>
{book_context}
</book_excerpts>

Score this single customer's interview for evidence quality: real specific pain, frequency, current workarounds, willingness to act.

SCORING (0-100): 75-100 GO, 50-74 PIVOT, 0-49 NO-GO.

Respond ONLY with valid JSON, no markdown fences:
{{
  "aggregate_score": <0-100>,
  "verdict": "<GO|PIVOT|NO-GO>",
  "verdict_reason": "<2 sentences>",
  "per_customer": [{{"name": "This customer", "score": <0-100>, "signal": "<strong|medium|weak>", "summary": "<1 sentence>"}}],
  "strong_signals": ["..."],
  "weak_signals": ["..."],
  "mom_test_violations": ["<answers that look like politeness, not evidence>"],
  "next_questions": ["..."],
  "recommendation": "<3-4 sentences>"
}}"""

    user_msg = f"""Product idea (NOT shown to customer): {req.product_idea}
Target: {req.target_customer} ({req.segment.upper()})
Problem hypothesis: {req.problem_hypothesis}

Interview:
{transcript}

Score this customer. Ground analysis in the book excerpts."""

    raw = call_groq(system_prompt, user_msg, max_tokens=1800)
    report = parse_json_object(raw)
    report["num_customers"] = 1
    report["book_excerpts_used"] = book_chunks[:2]
    return report
