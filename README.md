# Refund Agent

An agentic AI customer-support chatbot for e-commerce refunds, built with **LangGraph** and **LangChain**. It runs entirely on free tools: the Groq free-tier LLM and local HuggingFace embeddings.

## Features
- **Intent orchestration**: classifies each message (FAQ, refund, order status, human request, off-topic) plus sentiment, and routes it to the right handler
- **RAG**: answers policy and FAQ questions from `knowledge_base.md`, and escalates when no relevant section is found
- **Tool-using agent**: looks up orders, checks eligibility and issues refunds, with policy rules enforced in code rather than by the LLM
- **Human-in-the-loop**: refunds above ₹10,000 pause for supervisor approval; angry users or requests for a person create a support ticket
- **Guardrails**: PII redaction, prompt-injection blocking, and a check against false "refund processed" claims
- **Monitoring**: per-turn latency, per-node timing, token usage, tool calls, guardrail events and outcomes, logged to JSONL and shown on a dashboard

## Architecture
```
guard_in → classify → faq_rag | refund_agent → (human_review) | handoff | off_topic → guard_out
```

## Quick start
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key            # Windows PowerShell: $env:GROQ_API_KEY="your_key"
python refund_agent.py --demo           # scripted run covering every path
python refund_agent.py                  # interactive chat (/stats for metrics, exit to quit)
python monitor.py                       # monitoring dashboard
```

To use a different free LLM, set `MODEL`, for example `google_genai:gemini-2.5-flash` or `ollama:llama3.1`, and install the matching LangChain package.

## Files
| File | Purpose |
|---|---|
| `refund_agent.py` | LangGraph workflow, RAG, tools, guardrails, human-in-the-loop |
| `monitor.py` | Metrics collection and dashboard |
| `knowledge_base.md` | FAQ and refund policy used for RAG |

*A learning project. The order data is mocked.*
