# 🥒 Gherkin Chatbot

A dynamic chatbot API + web UI that turns natural-language requirements into Gherkin
syntax through conversation — it asks clarifying questions and refines iteratively.
Powered by the Gemini API free tier.

## API

### `POST /chat` — dynamic chatbot turn

Request:

```json
{
  "message": "As a member I want to apply a discount code at checkout",
  "history": [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hi! What should we convert today?"}
  ],
  "context": "E-commerce checkout; roles: guest, member"
}
```

- `message` (required): the user's new message.
- `history` (optional): prior turns, oldest first. Omit on the first turn.
- `context` (optional): persistent project context — domain terms, roles, constraints.

Response:

```json
{
  "reply": "Got it — a couple of quick questions…",
  "gherkin": "Feature: …\n  Scenario: …",
  "model": "gemini-3.8-flash"
}
```

- `reply`: the assistant's full reply (may be a clarifying question or contain Gherkin).
- `gherkin`: the Gherkin block extracted from the reply, or `null` if the assistant
  didn't produce Gherkin this turn (e.g. it asked a question instead).

### `POST /convert` — one-shot conversion (no history)

```json
{ "text": "Members can apply a discount code at checkout", "context": "E-commerce" }
```
→ `{ "gherkin": "Feature: …", "model": "gemini-3.8-flash" }`

### `GET /health`

`{"status": "ok", "model": "...", "key_configured": true}` — also tells you whether
the API key secret is set.

## Setup

1. Get a free Gemini API key at https://aistudio.google.com/app/apikey (Google account, no card).
2. Deploy: this repo includes `render.yaml` — connect it to Render (free tier) and it
   deploys as-is. Or run locally: `pip install -r requirements.txt && uvicorn app:app`.
3. Set the `GEMINI_API_KEY` environment variable / secret to your key.
4. Optional: override the model with `GEMINI_MODEL` (default `gemini-3.8-flash`).
