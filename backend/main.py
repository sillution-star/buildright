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

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
HF_API_KEY = os.environ["HF_API_KEY"]
HF_EMBEDDING_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 6

app = FastAPI(title="Mom Test Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_embedding(text: str) -> list[float]:
    resp = httpx.post(
        HF_EMBEDDING_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text},
        timeout=30,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding error: {resp.text}")
    result = resp.json()
    if result and isinstance(result[0], list):
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
    assumption: str = ""
    strong_answer: str = ""
    weak_answer: str = ""
    order: int = 0

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
    query_embedding = get_embedding(query)
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

    system_prompt = f"""You are a world-class customer discovery coach with deep knowledge of The Mom Test by Rob Fitzpatrick. You design structured interview guides that a non-expert field interviewer can follow top-to-bottom.

Relevant passages from The Mom Test book to ground your work:

<book_excerpts>
{book_context}
</book_excerpts>

STEP 1 — Read the product idea, the problem hypothesis, and the uploaded document. Identify TWO things and keep them strictly separate:
  (a) WHO the customer is — their identity/context (e.g. "owns a two-wheeler", "runs a small shop", "is a student"). This is ONLY background. Never interrogate the identity object itself.
  (b) THE PROBLEM TERRITORIES — the real areas of the customer's life and behavior that this specific product is betting on. DERIVE these yourself from the product and hypothesis. Every product has different territories; extract whatever THIS product's are. If the user's problem hypothesis names specific things to focus on, prioritise those.

Your questions MUST dig into the PROBLEM TERRITORIES (b), using identity (a) only as light context. Do NOT interrogate the identity object (e.g. if they own a bike but the product is about credit, do not ask about the bike — ask about their money behavior).

STEP 2 — Build a structured FUNNEL of {n} questions in this order:
  1. "Their World" — light context about the relevant part of their life (1-2 questions max)
  2. "Habits Today" — how they actually behave right now in the product's territory
  3. "Does the Problem Exist" — surface the real pain naturally, never assume it
  4. "Past Behavior" — concrete past episodes inside the problem territory (last time it happened, what they actually did)
  5. "Edge Cases" — the never-did-it person, the tried-and-gave-up person, the would-refuse person
  6. "Stakes" — how much this actually matters vs their other priorities

COVERAGE RULE — across the {n} questions, make sure you probe EVERY core assumption the product depends on (the bets that, if false, kill the product). Derive these assumptions from the product yourself. Spread questions so each major assumption is tested by at least one question.

THE MOM TEST RULES:
- You MUST ask about the customer's REAL history and behavior in the problem territory — their actual past actions, purchases, attempts, workarounds. This is their life; asking is required.
- What is FORBIDDEN is naming, describing, or pitching THE PRODUCT itself or its specific features/offer. Test the underlying behavior and need, never reveal the solution.

BANNED QUESTION TYPES — these fail the Mom Test. Do NOT produce any question like these:
- Opinion/importance questions: "How important is X to you?", "What do you think is the most important factor in X?", "What's the most frustrating part of X?" → these invite opinions, not facts. BANNED.
- Hypothetical/future questions: "What would you do if...?", "Would you ever...?", "How much could you afford to...?" → people lie about the future. BANNED.
- Product-revealing questions: anything naming a feature of the solution (e.g. "secured card", "locked deposit to get credit"). BANNED.
- Awareness/preference questions: "Have you ever considered X?", "What do you look for in X?" → BANNED.

REQUIRED QUESTION SHAPE — every question must be a SPECIFIC PAST EPISODE:
- Start most questions with "Tell me about the last time...", "Walk me through what happened when...", "The most recent time you... what did you do?"
- Each question must extract a real story, not a judgment or a guess.

NO REPETITION — this is critical:
- Every one of the {n} questions must probe a DISTINCT moment, behavior, or assumption. 
- Do NOT ask the same thing reworded (e.g. "frustrating part of managing finances" AND "frustrating part of using credit" AND "frustrating part of applying for credit" are the SAME question three times — NEVER do this).
- Before finalising, mentally check: would any two questions get essentially the same answer? If yes, replace one with a question probing a different assumption or a different past episode.

- For B2C: simple, conversational language an ordinary person speaks. For B2B: the language of their actual work.

STEP 3 — For EACH question provide a coaching card:
- "q": the exact question the interviewer speaks
- "category": one of "Their World" | "Habits Today" | "Does the Problem Exist" | "Past Behavior" | "Edge Cases" | "Stakes"
- "order": integer 1..{n} giving the sequence to ask them in
- "assumption": the SPECIFIC product bet this tests — concrete, never circular. (Good: "they will tolerate giving up X to get Y". Bad: "the customer has a need".)
- "strong_answer": a concrete example answer that would VALIDATE the assumption + why it's strong
- "weak_answer": a concrete example answer that would KILL the assumption + why it's a red flag
- "why": one line of Mom Test logic

Respond ONLY with a valid JSON array, ordered by "order". No preamble, no markdown fences.
Format:
[
  {{"q": "...", "category": "...", "order": 1, "assumption": "...", "strong_answer": "...", "weak_answer": "...", "why": "..."}}
]"""

    user_msg = f"""Build a structured {n}-question Mom Test interview guide.

Customer type: {req.segment.upper()}
Customer IDENTITY (background only, do NOT interrogate this): {req.target_customer}
Problem hypothesis / territories to focus on (steer your questions toward this): {req.problem_hypothesis}

UPLOADED DOCUMENT (product context — extract the real PROBLEM TERRITORIES and ASSUMPTIONS from here):
{req.additional_context[:6000] or 'None provided'}

The product being validated (NEVER reveal or hint at it): "{req.product_idea}"

Derive the problem territories and core assumptions from the product above — do not assume any fixed domain. Dig into the customer's real behavior in those territories, never their identity object, never the product itself. Ground every question in the book excerpts. Return ordered as a funnel.

FINAL CHECK before you respond: (1) every question is a specific PAST EPISODE, not an opinion or hypothetical; (2) no two questions would get the same answer; (3) no question names the product or its features. If any question fails, rewrite it."""

    raw = call_groq(system_prompt, user_msg, max_tokens=4000)
    data = parse_json_array(raw)
    # sort by order if present
    data.sort(key=lambda i: i.get("order", 0))
    questions = []
    for idx, i in enumerate(data, 1):
        # Fold the full coaching insight into `why` so it shows on screen without a frontend change
        insight_parts = []
        if i.get("assumption"):
            insight_parts.append(f"🎯 Tests: {i['assumption']}")
        if i.get("strong_answer"):
            insight_parts.append(f"✓ Strong: {i['strong_answer']}")
        if i.get("weak_answer"):
            insight_parts.append(f"✕ Red flag: {i['weak_answer']}")
        if i.get("why"):
            insight_parts.append(f"— {i['why']}")
        combined_why = "  ".join(insight_parts) if insight_parts else i.get("why", "")
        questions.append(Question(
            q=i["q"],
            why=combined_why,
            category=i.get("category", ""),
            assumption=i.get("assumption", ""),
            strong_answer=i.get("strong_answer", ""),
            weak_answer=i.get("weak_answer", ""),
            order=i.get("order", idx),
        ))
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
    headers = ["#", "Stage", "Question", "What it tests (assumption)", "✓ Strong answer sounds like", "✕ Red flag sounds like", "Weightage"] + [f"Customer {i+1}" for i in range(num_customers)]
    for col, h in enumerate(headers, start=1):
        c = ws.cell(row=header_row, column=col, value=h)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        c.fill = PatternFill("solid", start_color="1A1916")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    thin = Side(style="thin", color="DDDDDD")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    FIRST_CUST_COL = 8  # customer answers start at column 8

    for idx, q in enumerate(questions):
        r = header_row + 1 + idx
        ws.cell(row=r, column=1, value=q.get("order", idx + 1)).font = Font(name="Arial", size=10)
        ws.cell(row=r, column=2, value=q.get("category", "")).font = Font(name="Arial", size=9, color="666666")
        qc = ws.cell(row=r, column=3, value=q.get("q", ""))
        qc.font = Font(name="Arial", size=10, bold=True)
        qc.alignment = Alignment(wrap_text=True, vertical="top")
        ac = ws.cell(row=r, column=4, value=q.get("assumption", ""))
        ac.font = Font(name="Arial", size=9, color="555555")
        ac.alignment = Alignment(wrap_text=True, vertical="top")
        sc = ws.cell(row=r, column=5, value=q.get("strong_answer", ""))
        sc.font = Font(name="Arial", size=9, color="3B6D11")
        sc.alignment = Alignment(wrap_text=True, vertical="top")
        rc = ws.cell(row=r, column=6, value=q.get("weak_answer", ""))
        rc.font = Font(name="Arial", size=9, color="A32D2D")
        rc.alignment = Alignment(wrap_text=True, vertical="top")
        wc = ws.cell(row=r, column=7, value=3)
        wc.font = Font(name="Arial", size=10, color="0000FF")
        wc.alignment = Alignment(horizontal="center")
        wc.fill = PatternFill("solid", start_color="FFFDE7")
        for cust in range(num_customers):
            cell = ws.cell(row=r, column=FIRST_CUST_COL + cust, value="")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = border

    ws.column_dimensions["A"].width = 5
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 48
    ws.column_dimensions["D"].width = 38
    ws.column_dimensions["E"].width = 42
    ws.column_dimensions["F"].width = 42
    ws.column_dimensions["G"].width = 11
    for cust in range(num_customers):
        ws.column_dimensions[ws.cell(row=header_row, column=FIRST_CUST_COL + cust).column_letter].width = 40

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
        vals = [str(ws.cell(row=r, column=c).value) for c in range(1, 10)]
        if "Question" in vals:
            header_row = r
            break
    if header_row is None:
        raise HTTPException(status_code=422, detail="Could not find the question table. Use the exported template.")

    # Map columns by header name so layout changes never break this
    q_col = weight_col = None
    customer_cols = []
    col = 1
    while True:
        h = ws.cell(row=header_row, column=col).value
        if h is None and col > 30:
            break
        if h is None:
            col += 1
            if col > 60:
                break
            continue
        hs = str(h).strip()
        if hs == "Question":
            q_col = col
        elif hs == "Weightage":
            weight_col = col
        elif hs.lower().startswith("customer"):
            customer_cols.append((col, hs))
        col += 1
        if col > 60:
            break

    if q_col is None:
        raise HTTPException(status_code=422, detail="Could not find the Question column. Use the exported template.")
    if weight_col is None:
        weight_col = q_col + 4  # fallback

    rows = []
    r = header_row + 1
    while True:
        q = ws.cell(row=r, column=q_col).value
        if not q:
            break
        weight = ws.cell(row=r, column=weight_col).value or 3
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 3.0
        answers = {}
        for ccol, cname in customer_cols:
            ans = ws.cell(row=r, column=ccol).value
            if ans and str(ans).strip():
                answers[cname] = str(ans).strip()
        rows.append({"question": str(q), "weight": weight, "answers": answers})
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
