# Jarvis4 Demo Script

A conversational walkthrough that exercises every major module. Say each line after the wake word ("Hey Jarvis"). Expected responses are paraphrased — actual wording will vary.

---

## 1. Greeting & Personality (orchestrator, no tools)

> **You:** "Good morning Jarvis"
>
> **Expect:** A natural greeting. No tool use — the orchestrator handles this directly.
>
> *"Good morning, sir."*

---

## 2. Time Awareness (dynamic system prompt)

> **You:** "What time is it?"
>
> **Expect:** Current time, spoken naturally. Tests that the system prompt injects the live clock.
>
> *"It is 8:43 AM, sir."*

---

## 3. Weather (search agent → OpenWeatherMap MCP)

> **You:** "What's the weather like today?"
>
> **Expect:** Temperature, conditions, wind, sunrise/sunset. Delegated to search agent, which calls the weather MCP server.
>
> *"Sir, it is a crisp 27°F, clear sky, humidity around 44%, wind near 14 mph."*

---

## 4. Calendar (calendar agent → Google Calendar via gws)

> **You:** "What's on my calendar today?"
>
> **Expect:** Today's agenda or "nothing scheduled." Delegated to calendar agent, which runs `gws calendar`.
>
> *"Sir, at 6:10 PM you have TKD. No other engagements today."*

---

## 5. Email (mail agent → Gmail via gws)

> **You:** "Check my email, anything important?"
>
> **Expect:** Summary of recent/unread messages. Delegated to mail agent, which runs `gws gmail +triage`.
>
> *"One item from Meshy about a service retirement on March 20th."*

---

## 6. Home Control (home agent → Home Assistant REST API)

> **You:** "Turn on the living room lights"
>
> **Expect:** Confirmation. Delegated to home agent, which POSTs to HA.
>
> *"Consider it handled."*

---

## 7. Follow-up Context (conversation history)

> **You:** "Now turn them off"
>
> **Expect:** Understands "them" = living room lights from the previous turn. No need to repeat the room.
>
> *"Done. The living room lights are now off."*

---

## 8. Web Search (search agent → Brave Search MCP)

> **You:** "Search the web for the latest news about Cerebras"
>
> **Expect:** Current headlines synthesized into a spoken summary. Delegated to search agent, which calls Brave Search MCP.
>
> *"Sir, Cerebras is gearing up for an IPO in Q2 2026 after a $1 billion Series H..."*

---

## 9. Knowledge + Reasoning (orchestrator decides: direct or delegate)

> **You:** "What's the capital of Iceland and how far is it from here?"
>
> **Expect:** Factual answer (Reykjavik) plus distance estimate. May delegate to search or answer directly.
>
> *"The capital of Iceland is Reykjavik. From New York, roughly 2,600 miles."*

---

## 10. Slack (comms agent → Slack Web API)

> **You:** "Read the last few messages from the test Slack channel"
>
> **Expect:** Summary of recent messages from the channel. Delegated to comms agent.
>
> *"Five recent messages — a user joined, someone asked who won the latest F1 race..."*

---

## 11. Humor / Personality (orchestrator, no tools)

> **You:** "Tell me a joke"
>
> **Expect:** A dry, on-brand joke. No delegation — personality comes from the orchestrator.
>
> *"Why did the algorithm break up with the dataset? It found the relationship too... unstructured."*

---

## 12. Self-Awareness (orchestrator, no tools)

> **You:** "What can you do?"
>
> **Expect:** A summary of capabilities — lights, email, calendar, Slack, search, file ops. Shows the orchestrator knows its own toolset.
>
> *"I can toggle lights, adjust thermostats, send Slack or email messages, check your calendar..."*

---

## 13. Morning Briefing (multi-agent orchestration)

> **You:** "Give me a morning briefing"
>
> **Expect:** Weather + calendar + email + news in one spoken response. The orchestrator delegates to multiple agents and synthesizes their results.
>
> *"Sir, it is 8:45 AM. Outside: 27°F, clear. Your calendar shows TKD at 6:10. Unread mail flags a Navy Federal notice and a Meshy alert. In the news, Cerebras is preparing an IPO..."*

This is the most impressive demo — it fires off 3-4 agents and weaves the results into a single coherent briefing in under 10 seconds.

---

## Notes

- **Wake word:** Say "Hey Jarvis" before each line (or use the HTTP API for scripted demos).
- **Follow-ups:** After any spoken response, you have ~3 seconds to ask a follow-up without the wake word.
- **Speed:** Responses typically arrive in 1-3 seconds for simple queries, 5-10 for multi-agent tasks.
- **HTTP alternative:** `curl -X POST http://localhost:8787/chat -H "Content-Type: application/json" -d '{"text": "...", "speak": true}'`
