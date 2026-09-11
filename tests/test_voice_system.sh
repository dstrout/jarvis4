#!/usr/bin/env bash
# Jarvis4 Voice System Test — exercises each module via HTTP API.
#
# Usage:
#   ./tests/test_voice_system.sh              # Run all tests (text only)
#   ./tests/test_voice_system.sh --speak      # Run all tests with TTS playback
#   ./tests/test_voice_system.sh --test N     # Run specific test number
#
# Requires: Jarvis4 running on localhost:8787
#           curl, jq

set -euo pipefail

BASE="http://localhost:8787"
SPEAK="${SPEAK:-false}"
ONLY_TEST="${ONLY_TEST:-0}"
PASS=0
FAIL=0
SKIP=0

# Parse args
for arg in "$@"; do
    case "$arg" in
        --speak) SPEAK=true ;;
        --test)  shift; ONLY_TEST="${1:-0}" ;;
        [0-9]*)  ONLY_TEST="$arg" ;;
    esac
done

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- Helpers ---

chat() {
    local text="$1"
    local speak_flag="${2:-$SPEAK}"
    if [ "$speak_flag" = "true" ]; then
        curl -s -X POST "$BASE/chat" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$text\", \"speak\": true}" | jq -r '.response // empty'
    else
        curl -s -X POST "$BASE/chat" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$text\"}" | jq -r '.response // empty'
    fi
}

speak() {
    curl -s -X POST "$BASE/speak" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$1\"}" | jq -r '.status // .error // empty'
}

run_test() {
    local num="$1"
    local name="$2"
    local desc="$3"
    shift 3

    if [ "$ONLY_TEST" -ne 0 ] && [ "$ONLY_TEST" -ne "$num" ]; then
        return
    fi

    echo ""
    echo -e "${BLUE}━━━ Test $num: $name ━━━${NC}"
    echo -e "  ${desc}"
    echo ""

    # Run the test function
    if "$@"; then
        echo -e "  ${GREEN}✓ PASS${NC}"
        ((PASS++))
    else
        echo -e "  ${RED}✗ FAIL${NC}"
        ((FAIL++))
    fi
}

check_response() {
    local response="$1"
    local min_len="${2:-5}"

    if [ -z "$response" ]; then
        echo -e "  ${RED}Empty response${NC}"
        return 1
    fi

    if [ "${#response}" -lt "$min_len" ]; then
        echo -e "  ${RED}Response too short: '$response'${NC}"
        return 1
    fi

    # Truncate for display
    local display="$response"
    if [ "${#display}" -gt 200 ]; then
        display="${display:0:200}..."
    fi
    echo -e "  Response: ${display}"
    return 0
}

# --- Test Functions ---

test_http_status() {
    local result
    result=$(curl -s "$BASE/status" | jq -r '.service // empty')
    echo "  Service: $result"
    [ "$result" = "jarvis4" ]
}

test_direct_conversation() {
    local response
    response=$(chat "Hello, how are you doing today?")
    check_response "$response"
}

test_time_awareness() {
    local response
    response=$(chat "What time is it right now?")
    check_response "$response"
}

test_search_web() {
    local response
    response=$(chat "Search the web: what is the current population of Tokyo?")
    check_response "$response" 10
}

test_search_weather() {
    local response
    response=$(chat "What's the weather like in New York City right now?")
    check_response "$response" 10
}

test_email_check() {
    local response
    response=$(chat "Check my email inbox — just tell me how many unread messages I have")
    check_response "$response"
}

test_slack_read() {
    local response
    response=$(chat "Read the last 3 messages from the test_jarvis Slack channel")
    check_response "$response"
}

test_slack_send() {
    local response
    response=$(chat "Send a Slack message to the test_jarvis channel saying: Jarvis4 system test — ignore this message")
    check_response "$response"
}

test_home_control() {
    local response
    response=$(chat "What lights or devices are available in the house?")
    check_response "$response"
}

test_calendar() {
    local response
    response=$(chat "What's on my calendar for today?")
    check_response "$response"
}

test_system_tools() {
    local response
    response=$(chat "Use run_bash to run: echo 'Jarvis4 system test'")
    check_response "$response"
}

test_file_read() {
    local response
    response=$(chat "Read the first 5 lines of /home/ubuntu/work/Jarvis4/jarvis4.py")
    check_response "$response" 10
}

test_skill_lookup() {
    local response
    response=$(chat "Find skills related to gmail")
    check_response "$response"
}

test_tts_speak() {
    local result
    result=$(speak "Jarvis4 TTS test. The quick brown fox jumps over the lazy dog.")
    echo "  Status: $result"
    [ "$result" = "spoken" ]
}

test_alert() {
    local response
    response=$(curl -s -X POST "$BASE/alert" \
        -H "Content-Type: application/json" \
        -d '{"text": "System test alert — all modules operational"}' | jq -r '.response // empty')
    check_response "$response"
}

test_followup_context() {
    # Test multi-turn — first set context, then ask a follow-up
    chat "Remember the number 42 for this conversation" > /dev/null
    local response
    response=$(chat "What number did I just ask you to remember?")
    echo "  Response: $response"
    echo "$response" | grep -qi "42"
}

test_complex_delegation() {
    # Tests the orchestrator's ability to pick the right agent
    local response
    response=$(chat "What's the weather in London and do I have any meetings today? Give me a brief answer for both.")
    check_response "$response" 20
}

# --- Main ---

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Jarvis4 Voice System Test Suite        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Target: $BASE"
echo "  Speak:  $SPEAK"
echo ""

# Verify service is up
if ! curl -s "$BASE/status" > /dev/null 2>&1; then
    echo -e "${RED}Jarvis4 is not running at $BASE${NC}"
    echo "Start it with: sudo systemctl start jarvis4"
    exit 1
fi

# HTTP API
run_test  1 "HTTP /status"          "Health check endpoint"                                 test_http_status

# Orchestrator (direct conversation, no delegation)
run_test  2 "Direct conversation"   "Simple greeting — no tool use, orchestrator only"      test_direct_conversation
run_test  3 "Time awareness"        "Current time — tests dynamic system prompt"            test_time_awareness
run_test  4 "Multi-turn context"    "Remember + recall — conversation history"              test_followup_context

# Search agent (web + weather via MCP)
run_test  5 "Web search"            "Factual query → search agent → Brave MCP"             test_search_web
run_test  6 "Weather"               "Weather query → search agent → OpenWeatherMap MCP"    test_search_weather

# Email agent (Gmail via gws CLI)
run_test  7 "Email inbox"           "Check inbox → email agent → gws gmail +triage"        test_email_check

# Comms agent (Slack)
run_test  8 "Slack read"            "Read messages → comms agent → Slack API"              test_slack_read
run_test  9 "Slack send"            "Send message → comms agent → chat.postMessage"        test_slack_send

# Home control agent (Home Assistant)
run_test 10 "Home control"          "Device query → home agent → HA REST API"              test_home_control

# Calendar agent (Google Calendar via gws CLI)
run_test 11 "Calendar"              "Agenda check → calendar agent → gws calendar"         test_calendar

# System tools (bash, file I/O)
run_test 12 "System tools (bash)"   "Shell command → run_bash tool"                        test_system_tools
run_test 13 "File read"             "Read file → read_file tool"                           test_file_read

# Skills system
run_test 14 "Skill discovery"       "Find skills → find_skills tool → skill catalog"       test_skill_lookup

# TTS (direct speech, no LLM)
run_test 15 "TTS /speak"            "Direct TTS playback via /speak endpoint"              test_tts_speak

# Alert endpoint
run_test 16 "HTTP /alert"           "Alert → LLM + TTS via /alert endpoint"                test_alert

# Complex multi-agent
run_test 17 "Multi-agent query"     "Weather + calendar in one query — parallel delegation" test_complex_delegation

# Summary
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
TOTAL=$((PASS + FAIL))
echo -e "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC} / ${TOTAL} total"

if [ "$FAIL" -eq 0 ]; then
    echo -e "  ${GREEN}All tests passed!${NC}"
else
    echo -e "  ${YELLOW}Some tests failed — check output above${NC}"
fi
echo ""

exit "$FAIL"
