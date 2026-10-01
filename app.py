import os
import re

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = """You are a friendly, expert BDD assistant. Your job is to turn the user's \
natural-language requirements into Gherkin syntax through a dynamic conversation.

How to behave:
- Chat naturally and helpfully, like a real assistant — not a one-shot translator.
- If the user's description is vague, ambiguous, or missing details, ask 1-3 short \
clarifying questions instead of guessing.
- Refine iteratively: when the user answers or corrects you, update the Gherkin.
- When you have enough detail, output the Gherkin wrapped in a single ```gherkin code \
block. Only one Gherkin block per reply.
- Outside the code block you may explain briefly, summarize changes, or ask questions.
- Gherkin rules: start with a `Feature:` line; use `Scenario:` blocks (or \
`Scenario Outline:` with an `Examples:` table for data variations); use \
`Given` / `When` / `Then` with `And` / `But` continuations; keep steps declarative \
and implementation-agnostic; apply the project context below when it is provided."""

GHERKIN_FENCE = re.compile(r"```gherkin\s*(.*?)```", re.DOTALL | re.IGNORECASE)

app = FastAPI(title="Gherkin Chatbot", version="2.0.0")


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's new message")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation turns, oldest first",
    )
    context: str = Field(
        "",
        description="Optional persistent context: domain terms, roles, constraints",
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The assistant's full reply text")
    gherkin: str | None = Field(
        None, description="Gherkin extracted from the reply, if the assistant produced any"
    )
    model: str


class ConvertRequest(BaseModel):
    text: str = Field(..., description="Natural-language description of the behavior")
    context: str = Field("", description="Optional context: domain, roles, constraints")


class ConvertResponse(BaseModel):
    gherkin: str
    model: str


def _role_for_gemini(role: str) -> str:
    return "model" if role.strip().lower() in ("assistant", "model") else "user"


async def call_gemini(system_text: str, history: list[ChatMessage], message: str) -> str:
    """Send a dynamic chat turn to Gemini and return the raw reply text."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it as a secret in the Space settings.",
        )

    contents = [
        {"role": _role_for_gemini(m.role), "parts": [{"text": m.content}]}
        for m in history
        if m.content.strip()
    ]
    contents.append({"role": "user", "parts": [{"text": message}]})

    body = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.4},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(GEMINI_URL, json=body, headers=headers)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Gemini API: {exc}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Gemini API error {resp.status_code}: {resp.text[:500]}"
        )

    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail="Unexpected response from Gemini API.")


def extract_gherkin(reply: str) -> str | None:
    match = GHERKIN_FENCE.search(reply)
    return match.group(1).strip() if match else None


def build_system_text(context: str) -> str:
    if context.strip():
        return f"{SYSTEM_PROMPT}\n\nProject context:\n{context.strip()}"
    return SYSTEM_PROMPT


@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL, "key_configured": bool(GEMINI_API_KEY)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Dynamic chatbot turn: send a message + conversation history, get a reply.

    The assistant may ask clarifying questions or return Gherkin (also exposed
    separately in the `gherkin` field when present).
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="'message' must not be empty.")

    reply = await call_gemini(build_system_text(req.context), req.history, req.message)
    return ChatResponse(reply=reply, gherkin=extract_gherkin(reply), model=GEMINI_MODEL)


@app.post("/convert", response_model=ConvertResponse)
async def convert(req: ConvertRequest):
    """One-shot convenience endpoint: natural language -> Gherkin, no history."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty.")

    reply = await call_gemini(build_system_text(req.context), [], req.text)
    gherkin = extract_gherkin(reply)
    return ConvertResponse(gherkin=gherkin or reply, model=GEMINI_MODEL)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gherkin Chatbot</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
  #ctx { width: 100%; box-sizing: border-box; font-size: 0.95rem; }
  #log { border: 1px solid #ddd; border-radius: 8px; height: 380px; overflow-y: auto; padding: 1rem; margin: 1rem 0; background: #fafafa; }
  .msg { margin: 0.6rem 0; line-height: 1.45; }
  .msg.user { text-align: right; }
  .msg.user .bubble { display: inline-block; background: #e3f2fd; border-radius: 12px; padding: 0.5rem 0.9rem; text-align: left; max-width: 85%; }
  .msg.bot .bubble { display: inline-block; background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 0.5rem 0.9rem; max-width: 92%; }
  .msg pre { background: #f4f4f4; padding: 0.7rem; border-radius: 6px; white-space: pre-wrap; text-align: left; }
  #row { display: flex; gap: 0.5rem; }
  #inp { flex: 1; font-size: 1rem; padding: 0.5rem; }
  button { font-size: 1rem; padding: 0.5rem 1.25rem; cursor: pointer; }
  .err { color: #b00020; }
  small.hint { color: #666; }
</style>
</head>
<body>
<h1>&#129365; Gherkin Chatbot</h1>
<label for="ctx"><b>Context</b> <small class="hint">(optional — kept for the whole chat: domain terms, roles, constraints)</small></label><br>
<textarea id="ctx" rows="2" placeholder="e.g. E-commerce checkout; roles: guest, member"></textarea>
<div id="log"><div class="msg bot"><span class="bubble">Hi! Describe the behavior you want as Gherkin and I'll convert it — ask me to refine it as we go.</span></div></div>
<div id="row">
  <input id="inp" placeholder="Type your requirement…" onkeydown="if(event.key==='Enter')send()">
  <button onclick="send()">Send</button>
</div>
<script>
const history = [];
const log = document.getElementById('log');
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function renderBot(text){
  // Render ```gherkin blocks as <pre>, rest as escaped text with line breaks
  const parts = text.split(/```gherkin|```/);
  let html = '';
  parts.forEach((p, i) => {
    if (i % 2 === 1) html += '<pre>' + esc(p.replace(/^\\n+/,'')) + '</pre>';
    else html += esc(p).replace(/\\n/g, '<br>');
  });
  return html;
}
function addMsg(who, html){
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  d.innerHTML = '<span class="bubble">' + html + '</span>';
  log.appendChild(d); log.scrollTop = log.scrollHeight;
}
async function send(){
  const inp = document.getElementById('inp');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  addMsg('user', esc(text));
  history.push({role: 'user', content: text});
  addMsg('bot', '<i>thinking…</i>');
  const thinking = log.lastChild;
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, history: history.slice(0, -1),
                            context: document.getElementById('ctx').value})
    });
    const data = await r.json();
    thinking.remove();
    if (!r.ok) { addMsg('bot', '<span class="err">Error: ' + esc(data.detail || r.statusText) + '</span>'); history.pop(); return; }
    addMsg('bot', renderBot(data.reply));
    history.push({role: 'assistant', content: data.reply});
  } catch(e){ thinking.remove(); addMsg('bot', '<span class="err">Error: ' + esc(e.message) + '</span>'); history.pop(); }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML
