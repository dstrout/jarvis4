---
name: recipe-morning-briefing
version: 1.0.0
description: "Generate a morning briefing: top news headlines, local weather, and anything notable happening today."
metadata:
  openclaw:
    category: "recipe"
    domain: "information"
    requires:
      tools: ["web_search", "get_weather", "get_forecast"]
---

# Morning Briefing

Compile a concise morning briefing covering news, weather, and the day ahead.

## Steps

1. **Weather**: Get current weather and today's forecast for the user's location (default: Baltimore, MD).
2. **Top headlines**: Search for "top news today" to get 3-5 major headlines.
3. **Notable today**: Search for "what is happening today [date]" to find notable events, holidays, or observances.
4. **Synthesize**: Present as a spoken briefing:
   - Open with weather and how to dress
   - Key headlines (2-3 sentences each, max)
   - Anything notable about the day

## Tips

- Keep it under 60 seconds when spoken aloud — brevity is critical.
- Lead with weather since it affects immediate decisions (clothing, umbrella, commute).
- For headlines, focus on genuinely important news, not clickbait.
- If the user has previously mentioned interests, weight headlines accordingly.
