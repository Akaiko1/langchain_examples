import argparse
from collections import Counter
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "dist"}
_LANGUAGE_STATS_CACHE: Counter[str] | None = None  # reserved (not used)


def _clean_lines(lines: Iterable[str]) -> List[str]:
    cleaned = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        cleaned.append(line)
    return cleaned


def _run_rg(query: str, max_hits: int) -> Tuple[int, str]:
    command = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color",
        "never",
        "--max-count",
        str(max_hits),
        query,
        ".",
    ]
    proc = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout if proc.stdout else proc.stderr


def _python_search(query: str, max_hits: int) -> str:
    hits = []
    allowed_suffixes = {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
    }
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir():
            if path.name in _SKIP_DIRS or path.name.startswith("."):
                continue
            continue
        if any(part in _SKIP_DIRS or part.startswith(".") for part in path.parts):
            continue
        if path.suffix and path.suffix.lower() not in allowed_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if query.lower() in line.lower():
                rel = path.relative_to(REPO_ROOT)
                snippet = line.strip()
                hits.append(f"{rel}:{idx}: {snippet}")
                if max_hits and len(hits) >= max_hits:
                    return "\n".join(hits)
    if not hits:
        return "No matches found."
    return "\n".join(hits)


def _repo_search_impl(query: str) -> str:
    query = (query or "").strip()
    if len(query) < 2:
        return "Please provide a longer search query (2+ characters)."

    # Configurable cap to avoid overwhelming outputs; set REPO_SEARCH_MAX_HITS=0 to disable
    try:
        max_hits = int(os.getenv("REPO_SEARCH_MAX_HITS", "80"))
    except ValueError:
        max_hits = 80
    if shutil.which("rg"):
        code, output = _run_rg(query, max_hits if max_hits > 0 else 0)
        if code in {0, 1} and output:
            lines = _clean_lines(output.splitlines())
            if not lines:
                return "No matches found."
            return "\n".join(lines if max_hits == 0 else lines[:max_hits])
        # fall back if rg failed unexpectedly
    return _python_search(query, max_hits)


@tool
def repo_search(query: str) -> str:
    """Search the repository for a string and return up to 20 matches with file and line numbers."""
    return _repo_search_impl(query)


def _list_repo_impl(path: str = "") -> str:
    rel = (path or ".").strip()
    target = (REPO_ROOT / rel).resolve()
    try:
        target.relative_to(REPO_ROOT)
    except ValueError:
        return "Path outside repository."
    if not target.exists():
        return "Path not found."
    if target.is_file():
        return f"{target.relative_to(REPO_ROOT)} (file)"

    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    if not entries:
        return "(empty directory)"
    parts = []
    for entry in entries[:200]:
        if entry.name.startswith("."):
            continue
        suffix = "/" if entry.is_dir() else ""
        parts.append(str(entry.relative_to(REPO_ROOT)) + suffix)
    return "\n".join(parts)


@tool
def list_repo(path: str = "") -> str:
    """List folders and files at the repo root or a relative path (directories suffixed with /)."""
    return _list_repo_impl(path)


@tool
def count_occurrences(term: str) -> str:
    """Count case-sensitive occurrences of a string across the repository."""
    term_clean = (term or "").strip()
    if not term_clean:
        return "Please provide a non-empty term."
    count = _count_term_occurrences(term_clean)
    return f'"{term_clean}" occurs {count} times (case-sensitive).'



# Removed heuristic fallbacks that inferred needs from question text; the agent now decides.


def _normalize_tool_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip()
    if "(" in name:
        name = name.split("(", 1)[0]
    if " " in name:
        name = name.split(" ", 1)[0]
    return name.strip()


def _clean_tool_input(value) -> str:
    if isinstance(value, str):
        return value.strip().strip('"')
    return str(value).strip()


def _summarize_observation(tool: str, observation: str, question: str | None = None) -> str:
    if not isinstance(observation, str):
        observation = str(observation)
    obs = observation.strip()
    if not obs:
        return ""

    question_hint = f" relative to question '{question.strip()}'" if question else ""

    if tool == "repo_search":
        lines = [line.strip() for line in obs.splitlines() if line.strip()]
        if not lines:
            return "No matches found."
        previews = []
        for line in lines[:3]:
            parts = line.split(":", 2)
            if len(parts) >= 3:
                loc = f"{parts[0]}:{parts[1]}"
                snippet = parts[2].strip()
                previews.append(f"{loc} → {snippet[:80]}")
            else:
                previews.append(line[:100])
        joined = "; ".join(previews)
        extra = " (additional matches omitted)" if len(lines) > 3 else ""
        return f"Key repo references{question_hint}: {joined}{extra}."

    if tool == "list_repo":
        entries = [line.strip() for line in obs.splitlines() if line.strip()]
        if not entries:
            return "Directory listing empty."
        folders = [e.rstrip("/") for e in entries if e.endswith("/")]
        files = [e for e in entries if not e.endswith("/")]
        summary_parts = []
        if folders:
            summary_parts.append("folders: " + ", ".join(folders[:5]))
            if len(folders) > 5:
                summary_parts[-1] += " …"
        if files:
            summary_parts.append("files: " + ", ".join(files[:3]))
            if len(files) > 3:
                summary_parts[-1] += " …"
        summary = "; ".join(summary_parts)
        return f"Repo layout glimpse{question_hint}: {summary}."

    if tool == "count_occurrences":
        return obs

    return obs[:200]


def _extract_summary(observation: str) -> str:
    if not isinstance(observation, str):
        return ""
    marker = "Summary:"
    idx = observation.rfind(marker)
    if idx == -1:
        return ""
    summary = observation[idx + len(marker):].strip()
    return summary


# Removed model-driven fallback synthesis; rely on agent's own reasoning instead.


def _count_term_occurrences(term: str) -> int:
    if not term:
        return 0
    term = term.strip()
    if shutil.which("rg"):
        command = [
            "rg",
            "--no-heading",
            "--count-matches",
            term,
            ".",
        ]
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if proc.returncode in {0, 1}:
            total = 0
            for line in proc.stdout.splitlines():
                try:
                    count = int(line.rsplit(":", 1)[-1])
                except (ValueError, IndexError):
                    continue
                total += count
            return total

    # Fallback: manual scan (case-sensitive)
    total = 0
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir():
            if path.name in _SKIP_DIRS or path.name.startswith("."):
                continue
            continue
        if any(part in _SKIP_DIRS or part.startswith(".") for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        total += text.count(term)
    return total


# Removed language detection heuristic; agent can perform any needed checks via tools.


# Removed language stats helpers; agent can derive such info via tools if needed.


class NormalizingAgentExecutor(AgentExecutor):
    def _reset_history(self) -> None:
        object.__setattr__(self, "_tool_history", set())
        object.__setattr__(self, "_should_stop_now", False)

    def _normalize_agent_action(self, agent_action: AgentAction) -> AgentAction:
        if not isinstance(agent_action, AgentAction):
            return agent_action
        cleaned = _normalize_tool_name(agent_action.tool)
        if cleaned == agent_action.tool:
            return agent_action
        return agent_action.copy(update={"tool": cleaned})

    def _remember(self, tool: str, tool_input: str) -> bool:
        history = getattr(self, "_tool_history", None)
        if history is None:
            self._reset_history()
            history = getattr(self, "_tool_history", None)
        key = (tool, tool_input)
        if history is not None and key in history:
            return True
        if history is None:
            history = set()
            object.__setattr__(self, "_tool_history", history)
        history.add(key)
        return False

    def _enrich_observation(self, observation, repeated: bool, *, tool: str):
        question = getattr(self, "_current_question", None)
        summary = _summarize_observation(tool, observation, question)
        enriched = observation if isinstance(observation, str) else str(observation)
        if summary and summary not in enriched:
            enriched = f"{enriched}\nSummary: {summary}"
        if repeated:
            enriched = (
                f"{enriched}\n(You've already retrieved this result with the same tool input. Reflect on it and finish with `Thought:` then `Final Answer`.)"
            )
        return enriched

    def _perform_agent_action(
        self,
        name_to_tool_map,
        color_mapping,
        agent_action,
        run_manager=None,
    ):
        normalized_action = self._normalize_agent_action(agent_action)
        repeated = self._remember(
            normalized_action.tool,
            _clean_tool_input(getattr(normalized_action, "tool_input", ""))
        )
        step = super()._perform_agent_action(
            name_to_tool_map,
            color_mapping,
            normalized_action,
            run_manager=run_manager,
        )
        enriched = self._enrich_observation(
            step.observation, repeated, tool=normalized_action.tool
        )
        if repeated:
            object.__setattr__(self, "_should_stop_now", True)
        return step.__class__(action=step.action, observation=enriched)

    async def _aperform_agent_action(
        self,
        name_to_tool_map,
        color_mapping,
        agent_action,
        run_manager=None,
    ):
        normalized_action = self._normalize_agent_action(agent_action)
        repeated = self._remember(
            normalized_action.tool,
            _clean_tool_input(getattr(normalized_action, "tool_input", ""))
        )
        step = await super()._aperform_agent_action(
            name_to_tool_map,
            color_mapping,
            normalized_action,
            run_manager=run_manager,
        )
        enriched = self._enrich_observation(
            step.observation, repeated, tool=normalized_action.tool
        )
        if repeated:
            object.__setattr__(self, "_should_stop_now", True)
        return step.__class__(action=step.action, observation=enriched)

    def _get_tool_return(self, next_step_output):
        agent_action, observation = next_step_output
        normalized_action = self._normalize_agent_action(agent_action)
        return super()._get_tool_return((normalized_action, observation))

    def _call(self, inputs, run_manager=None):
        self._reset_history()
        object.__setattr__(self, "_current_question", inputs.get("input"))
        try:
            return super()._call(inputs, run_manager=run_manager)
        finally:
            object.__setattr__(self, "_tool_history", None)
            object.__setattr__(self, "_should_stop_now", False)
            object.__setattr__(self, "_current_question", None)

    async def _acall(self, inputs, run_manager=None):
        self._reset_history()
        object.__setattr__(self, "_current_question", inputs.get("input"))
        try:
            return await super()._acall(inputs, run_manager=run_manager)
        finally:
            object.__setattr__(self, "_tool_history", None)
            object.__setattr__(self, "_should_stop_now", False)
            object.__setattr__(self, "_current_question", None)

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if getattr(self, "_should_stop_now", False):
            return False
        return super()._should_continue(iterations, time_elapsed)

# Removed path-reference and tool-usage helpers tied to fallback heuristics.


def postprocess_answer(_question: str, result: dict) -> str:
    return result.get("output", "(no output)")


def _format_intermediate_steps(steps) -> str:
    if not steps:
        return ""
    blocks = []
    for idx, (action, observation) in enumerate(steps, start=1):
        lines = [f"Step {idx}:"]
        log = getattr(action, "log", "") if action else ""
        # Extract an explicit Thought line if present in the log
        if isinstance(log, str) and log:
            for raw in log.splitlines():
                s = raw.strip()
                if s.lower().startswith("thought:"):
                    lines.append(s)
                    break
        tool_name = getattr(action, "tool", "") if action else ""
        tool_input = getattr(action, "tool_input", "") if action else ""
        if tool_name:
            lines.append(f"Tool: {tool_name}")
        if tool_input:
            lines.append(f"Tool Input: {tool_input}")
        if observation is not None:
            obs_str = observation if isinstance(observation, str) else str(observation)
            lines.append(f"Observation: {obs_str.strip()}")
            # Also surface a concise summary (if present) as an explicit agent reflection
            summary = _extract_summary(obs_str)
            if summary:
                lines.append(f"Agent Summary: {summary}")
            elif not any(l.lower().startswith("thought:") for l in lines):
                # Synthesize a brief thought from the observation when none present
                synth = obs_str.strip().splitlines()[-1][:140]
                if synth:
                    lines.append(f"Thought (synth): {synth}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _parsing_error_handler(_error: Exception) -> str:
    return (
        "Invalid action format. Respond with EXACTLY one of the following:\n"
        "1) Thought: <reasoning>\nAction: <tool_name>\nAction Input: \"...\"\n"
        "OR\n2) Thought: <reasoning>\nFinal Answer: <concise answer>."
    )


def build_agent(verbose: bool = False) -> AgentExecutor:
    llm = ChatOllama(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
    tools = [repo_search, list_repo, count_occurrences]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                textwrap.dedent(
                    """
                    You are a careful coding assistant working inside this repository.\n
                    You have access to the following tools:\n{tools}\n
                    Prefer to call tools when they can ground your answer—use `repo_search` for definitions, `list_repo` for structure, and `count_occurrences` for literal frequency checks.\n
                    If a tool response is empty or unclear, acknowledge the uncertainty rather than guessing.\n
                    Use the ReAct format to decide when to call tools and when to respond directly.\n
                    After every observation, write a fresh `Thought:` summarizing what you learned and whether it already answers the question. Only call another tool if the summaries are insufficient.\n
                    Always begin each message with `Thought:` and, once ready to respond, finish with a `Final Answer:` line.\n
                    When you choose an action, follow this format exactly:
                    Action: tool_name  (just the bare tool name, no parentheses or arguments)
                    Action Input: "..." (use double quotes, even for an empty string).\n
                    Never write things like `Action: repo_search("foo")`, `Action: repo_search(where)`, or include extra words after the tool name—those are invalid formats.\n                    Correct example:\n                    Action: repo_search\n                    Action Input: "main"\n                    Incorrect (never do this):\n                    Action: repo_search("main")\n                    Action Input: main\n                    Action: repo_search where\n                    If you ever see an observation that says a tool name is invalid, immediately retry with the correct format above.\n                    Avoid calling the same tool repeatedly with the same input once you have the information you need.\n
                    Example:\n                    Thought: I should inspect the repository layout.\n                    Action: list_repo\n                    Action Input: ""\n                    Observation: Agents/\n                    Observation: RAG/\n                    Observation: README.md\n                    Observation: Workflows/\n                    Observation: wiki/\n                    Thought: I can summarize the top-level folders now.\n                    Final Answer: Top-level entries: Agents/, RAG/, Workflows/, wiki/, plus root files such as README.md.\n
                    Follow this protocol:\n                    1. Think by writing a `Thought:` line.\n                    2. When you need information from a tool, write an `Action:` line with one tool name from {tool_names}.\n                       Then provide the tool input on a new line prefixed with `Action Input:`.\n                    3. After the tool returns, note its result on an `Observation:` line.\n                    4. Repeat Thought/Action/Observation as needed.\n                    5. When you have the answer, write `Thought: I now know the final answer.` followed by\n                       `Final Answer:` on its own line with a concise response that cites filenames when relevant.\n
                    Do not hallucinate file names. Prefer the search tool over guessing. Keep answers short.
                    """
                ).strip(),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Question: {input}\n\nBegin!",
            ),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    agent = create_react_agent(llm, tools, prompt)
    executor = NormalizingAgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=_parsing_error_handler,
        max_iterations=8,
        return_intermediate_steps=True,
    )
    return executor


def interactive_loop(
    executor: AgentExecutor,
    *,
    verbose: bool = False,
    show_steps: bool = False,
) -> None:
    chat_history: List[BaseMessage] = []
    print("Terminal agent ready. Type your question (or 'exit' to quit).\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            print("Bye.")
            break
        result = executor.invoke({"input": user_input, "chat_history": chat_history})
        if show_steps:
            steps_text = _format_intermediate_steps(result.get("intermediate_steps"))
            if steps_text:
                print(f"\n{steps_text}\n")
        output = postprocess_answer(user_input, result)
        print(f"\n{output}\n")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))


def run_once(
    executor: AgentExecutor,
    question: str,
    *,
    show_steps: bool = False,
) -> None:
    result = executor.invoke({"input": question, "chat_history": []})
    if show_steps:
        steps_text = _format_intermediate_steps(result.get("intermediate_steps"))
        if steps_text:
            print(steps_text)
            print("")
    print(postprocess_answer(question, result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple LangChain agent with repo search tool.")
    parser.add_argument("--question", "-q", help="Ask a single question and exit.")
    parser.add_argument("--verbose", action="store_true", help="Print agent trace events.")
    parser.add_argument(
        "--hide-steps",
        action="store_true",
        help="Do not print intermediate tool calls and observations.",
    )
    # No auto-insights flag; agent fully drives reasoning now
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    executor = build_agent(verbose=args.verbose)

    show_steps = not args.hide_steps
    # No auto-insights: always rely on agent output

    if args.question:
        run_once(executor, args.question, show_steps=show_steps)
    else:
        interactive_loop(executor, verbose=args.verbose, show_steps=show_steps)


if __name__ == "__main__":
    main()
