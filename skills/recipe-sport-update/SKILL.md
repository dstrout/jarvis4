---
name: recipe-sport-update
version: 1.0.0
description: "Get a comprehensive update on a sport or league: standings, recent results, headlines, and next event with weather."
metadata:
  openclaw:
    category: "recipe"
    domain: "information"
    requires:
      tools: ["web_search", "get_forecast"]
---

# Sport / League Update

Get a comprehensive update on any sport or racing series (F1, NFL, NBA, Premier League, etc).

## Steps

1. **Current standings**: Search for "[league] [current year] standings" to get the current championship or league standings.
2. **Recent results**: Search for "[league] latest results" to find the most recent race, match, or game results.
3. **Headlines**: Search for "[league] news" to find any major headlines — transfers, controversies, injuries, rule changes.
4. **Next event**: Search for "[league] next race/match/game schedule" to find the upcoming event, including date, time, and location.
5. **Weather at next event** (if outdoor sport): Use get_forecast for the location and date of the next event.
6. **Synthesize**: Combine all findings into a concise briefing. Lead with the most interesting or surprising information.

## Tips

- Adjust terminology per sport: "race" for F1/NASCAR, "match" for soccer/tennis, "game" for NFL/NBA.
- For motorsport, include qualifying results if the next event is imminent.
- If standings are mid-season, note how many events remain.
- Skip the weather step for indoor sports.
