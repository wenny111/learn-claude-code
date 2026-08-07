#!/usr/bin/env python3
"""
s08_context_compact.py - Context Compact

Four-layer compaction pipeline inserted before LLM calls:

    L1: snip_compact      — trim middle messages when count > 50
    L2: micro_compact     — replace old tool_results with placeholders
    L3: tool_result_budget — persist large results to disk
    L4: compact_history   — LLM full summary (1 API call)

    Emergency: reactive_compact — when API still returns prompt_too_long

    ┌─────────────────────────────────────────────────────────────┐
    │  messages[]                                                 │
    │    ↓                                                        │
    │  L3 budget ─→ L1 snip ─→ L2 micro ─→ [token > threshold?]  │
    │                                      ├─ No  → LLM          │
    │                                      └─ Yes → L4 summary   │
    │                                              ↓              │
    │                                          LLM call           │
    │                                    [prompt_too_long?]        │
    │                                      └─ Yes → reactive      │
    └─────────────────────────────────────────────────────────────┘

Core principle: cheap first, expensive last.
Execution order matches CC source: budget → snip → micro → auto.

Builds on s07 (skill loading). Usage:

    python s08_context_compact/code.py
    Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""

import ast, json, locale, os, subprocess, time
from pathlib import Path

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"): os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd().resolve()
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
CURRENT_TODOS: list[dict] = []

def _decode_text(data: bytes | None) -> str:
    """Decode tool output without depending on the host's default encoding."""
    if not data:
        return ""
    encodings = ("utf-8-sig", locale.getpreferredencoding(False))
    for encoding in dict.fromkeys(encodings):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            pass
    return data.decode("utf-8", errors="replace")

def _read_text(path: Path) -> str:
    return _decode_text(path.read_bytes())

# s07: Skill catalog scan (inherited from s07)
def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()

SKILL_REGISTRY: dict[str, dict] = {}

def _scan_skills():
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = _read_text(manifest)
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}

_scan_skills()

def list_skills() -> str:
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]

# s08: SYSTEM includes skill catalog (inherited from s07 build_system)
def build_system() -> str:
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )

SYSTEM = build_system()

# s08: subagent gets its own system prompt — no compact, no skill loading
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s07 (unchanged): Basic Tools
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, timeout=120)
        out = (_decode_text(r.stdout) + _decode_text(r.stderr)).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired: return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = _read_text(safe_path(path)).splitlines()
        if limit and limit < len(lines): lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path); file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8"); return f"Wrote {len(content)} bytes to {path}"
    except Exception as e: return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = _read_text(file_path)
        if old_text not in text: return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e: return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e: return f"Error: {e}"

def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None

def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

def extract_text(content) -> str:
    if not isinstance(content, list): return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


# ═══════════════════════════════════════════════════════════
#  FROM s06-s07 (unchanged): Subagent
# ═══════════════════════════════════════════════════════════

SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]
SUB_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write,
                "edit_file": run_edit, "glob": run_glob}

def spawn_subagent(description: str) -> str:
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]
    for _ in range(30):
        response = client.messages.create(model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": str(blocked)})
                    continue
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                trigger_hooks("PostToolUse", block, output)
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
    result = extract_text(messages[-1]["content"])
    if not result:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result


# ═══════════════════════════════════════════════════════════
#  NEW in s08: Four-Layer Compaction Pipeline
# ═══════════════════════════════════════════════════════════

CONTEXT_LIMIT = 50000      # 轻量压缩后仍超过此字符数时，触发 L4 全量摘要
KEEP_RECENT = 3            # L2 保留最近 3 个 tool_result 原文，其余旧结果可替换为占位符
PERSIST_THRESHOLD = 30000  # L3 中单条结果超过此字符数时，才有资格落盘并改为路径 + 预览

# 教学版用字符数近似上下文大小，避免引入 tokenizer；生产系统应按模型的
# token 计数和实际请求开销来判断预算。
def estimate_size(msgs): return len(str(msgs))

def _block_type(block):
    # Anthropic 返回的 block 是 SDK 对象，我们追加的 block 常是 dict；
    # 分别用属性访问和字典访问，统一取得 type。
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _message_has_tool_use(msg):
    # assistant 的 tool_use 必须和紧随其后的 user/tool_result 成对保留；
    # 否则截断历史可能留下没有结果的工具请求，破坏对话协议。
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(_block_type(block) == "tool_use" for block in content)


def _is_tool_result_message(msg):
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result"
               for block in content)


# L1: snipCompact — trim middle messages
def snip_compact(messages, max_messages=50):
    # 第 1 步：判断是否需要裁剪。未超过上限时直接返回原列表。
    if len(messages) <= max_messages: return messages

    # 第 2 步：计算头尾保留范围。默认保留头 3 条、尾 47 条。
    keep_head, keep_tail = 3, max_messages - 3

    # 第 3 步：计算切片边界。
    # head_end=3 对应 messages[:3]；tail_start 是尾部切片的起始索引。
    head_end, tail_start = keep_head, len(messages) - keep_tail

    # 第 4 步：检查头部边界，避免保留 tool_use 却删掉其 tool_result。
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        # 扩大头部范围，把紧随其后的 tool_result 一起保留。
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1

    # 第 5 步：检查尾部边界，避免保留 tool_result 却删掉其 tool_use。
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        # 尾部起点向前移动一条，把对应的 tool_use 一起保留。
        tail_start -= 1

    # 调整后若头尾已经重叠，说明没有可安全删除的中间区域。
    if head_end >= tail_start:
        return messages

    # 第 6 步：计算删除数量并拼出新列表。
    # messages[:head_end] 取头部，messages[tail_start:] 取尾部；
    # 中间只留一条提示，告诉模型有多少条消息被删除。
    snipped = tail_start - head_end
    return messages[:head_end] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[tail_start:]


# L2: microCompact — old result placeholders
def collect_tool_results(messages):
    # 第 1 步：遍历全部消息，收集每个 tool_result。
    # enumerate(...) 同时返回“索引、元素”；block 是原字典的引用。
    blocks = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list): continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks

def micro_compact(messages):
    # 第 2 步：取得全部 tool_result；不超过 KEEP_RECENT 个时无需压缩。
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT: return messages

    # 第 3 步：[:-KEEP_RECENT] 表示“除最后 3 个之外的所有旧结果”。
    # 最近 3 个保留原文，旧结果只有超过 120 字符才替换。
    for _, _, block in tool_results[:-KEEP_RECENT]:
        if len(block.get("content", "")) > 120:
            # 直接修改原 block 的 content；type 和 tool_use_id 保持不变。
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


# L3: toolResultBudget — 把大结果写回磁盘，返回路径 + 预览
def persist_large_output(tool_use_id, output):
    # 第 1 步：单条结果未超过落盘门槛时原样返回。
    if len(output) <= PERSIST_THRESHOLD: return output

    # 第 2 步：创建输出目录，用 tool_use_id 作为文件名保存完整结果。
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists(): path.write_text(output, encoding="utf-8")

    # 第 3 步：返回“文件路径 + 前 2000 字符预览”，替代上下文中的完整结果。
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"

# L3：判断最新一批结果是否超过总预算，并选择需要落盘的结果。
def tool_result_budget(messages, max_bytes=200_000):
    # 第 1 步：取最后一条消息；空列表或结构不符合时不处理。
    last = messages[-1] if messages else None
    if not last or last.get("role") != "user" or not isinstance(last.get("content"), list): return messages

    # 第 2 步：用列表推导式筛出最后一条消息中的所有 tool_result。
    # enumerate(...) 产生 (索引, 元素)，所以 blocks 每项也是 (索引, block)。
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]

    # 第 3 步：计算这一批结果的总字符数；未超过 200000 时原样返回。
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes: return messages

    # 第 4 步：按单条结果长度从大到小排序，优先落盘最大的结果。
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)

    # 第 5 步：逐条替换，直到总大小回到预算以内。
    for _, block in ranked:
        if total <= max_bytes: break
        content = str(block.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD: continue
        tid = block.get("tool_use_id", "unknown")
        # block 是 messages 中的原字典，因此这里会原地替换 content。
        block["content"] = persist_large_output(tid, content)
        # 每替换一条都重新计算总大小，满足预算后下一轮会 break。
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


# L4: autoCompact — LLM full summary
def write_transcript(messages):
    # 第 1 步：将当前历史逐条写成 JSONL，保留磁盘审计副本。
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for msg in messages: f.write(json.dumps(msg, default=str) + "\n")
    return path

# L4：这是唯一会额外调用模型的压缩层。
def summarize_history(messages):
    # 第 2 步：把消息序列化为文本；[:80000] 只取前 80000 个字符。
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    # 第 3 步：额外调用一次模型，生成目标、决策和剩余工作的摘要。
    response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    return "\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text").strip() or "(empty summary)"

def compact_history(messages):
    # 第 4 步：先保存当前历史，再取得模型摘要。
    # transcript 只用于磁盘审计；教学版不会自动把它重新读给模型。
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)

    # 第 5 步：返回只含一条摘要的新列表，调用方再用 messages[:] 替换历史。
    # system prompt 和工具定义不在 messages 中，下一轮仍会单独发送。
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# Emergency: reactiveCompact — on API error
def reactive_compact(messages):
    # 第 1 步：API 已报告上下文过长时，先保存当前历史。
    transcript = write_transcript(messages)

    # 第 2 步：默认保留最后 5 条消息；tail_start 是尾部起始索引。
    tail_start = max(0, len(messages) - 5)

    # 第 3 步：若尾部从 tool_result 开始，再向前保留对应的 tool_use。
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1

    # 第 4 步：只总结较早历史，最近消息保持原文。
    summary = summarize_history(messages[:tail_start])

    # *messages[tail_start:] 会把尾部列表的元素逐个展开到新列表中。
    # 最终结果是“一条早期摘要 + 最近 5 条左右的原始消息”。
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]


# ═══════════════════════════════════════════════════════════
#  FROM s07: Tool Definitions
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "todo_write", "description": "Create and manage a task list for your current coding session.",
     "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}},
    {"name": "task", "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
    {"name": "load_skill", "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    # 这是模型主动触发 L4 的入口。focus 在教学版 schema 中保留，
    # 但当前 compact_history 尚未使用它来定制摘要重点。
    {"name": "compact", "description": "Summarize earlier conversation to free context space.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
    "task": spawn_subagent, "load_skill": load_skill,
}

# FROM s04 (unchanged): Hooks
HOOKS = {"PreToolUse": [], "PostToolUse": []}
def trigger_hooks(event, *args):
    for cb in HOOKS[event]:
        r = cb(*args)
        if r is not None: return r
    return None

DENY_LIST = ["rm -rf /", "sudo", "shutdown"]
def permission_hook(block):
    if block.name == "bash":
        for p in DENY_LIST:
            if p in block.input.get("command", ""): return "Permission denied"
    return None
def log_hook(block):
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None

HOOKS["PreToolUse"].append(permission_hook)
HOOKS["PreToolUse"].append(log_hook)


# ═══════════════════════════════════════════════════════════
#  agent_loop — s08 core: run compaction pipeline before LLM
# ═══════════════════════════════════════════════════════════

MAX_REACTIVE_RETRIES = 1  # retry limit for reactive compact

def agent_loop(messages: list):
    reactive_retries = 0
    while True:
        # 每次请求模型前都先跑无额外 API 调用的三层清理。
        # 顺序是 budget → snip → micro：先处理刚产生的超大输出，
        # 再裁剪历史消息，最后缩短仍保留的旧工具结果。
        # messages[:] 是全切片赋值：替换全部元素但保留列表对象本身。
        # 这样调用方传入的 history 仍指向同一列表；若写 messages = ...，
        # 只会让当前函数的局部变量改指向，外层 history 不会同步更新。
        messages[:] = tool_result_budget(messages)  # L3: 存大结果
        messages[:] = snip_compact(messages)     # L1：删中间
        messages[:] = micro_compact(messages) # L2：删旧结果

        # 轻量压缩后仍超过预算，才付出一次额外模型调用做 L4 全量摘要。
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        try:
            response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
            reactive_retries = 0  # reset on successful API call
        except Exception as e:
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                # 应急压缩保留最近消息，continue 后用缩短后的同一历史重新请求。
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue
            raise

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use": return

        results = []
        for block in response.content:
            if block.type != "tool_use": continue
            print(f"\033[36m> {block.name}\033[0m")

            # compact 不是普通 handler：它会替换 messages 历史，因此要立即
            # 结束本轮工具分发，下一次循环用压缩后的上下文重新请求模型。
            if block.name == "compact":
                # 主动 compact 与自动 L4 一样，也要原地替换共享的历史列表。
                messages[:] = compact_history(messages)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compacted. Conversation history has been summarized.]"})
                messages.append({"role": "user", "content": results})
                break  # end current turn, start fresh with compacted context

            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(blocked)})
                continue
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            trigger_hooks("PostToolUse", block, output)
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        else:
            # Python 的 for...else 仅在循环没有 break 时执行：
            # 正常工具路径在这里统一追加结果；compact 路径已在上面追加。
            messages.append({"role": "user", "content": results})
            continue
        # compact was called: results already appended above
        continue


if __name__ == "__main__":
    print("s08: Context Compact — four-layer compaction pipeline")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try: query = input("\033[36ms08 >> \033[0m")
        except (EOFError, KeyboardInterrupt): break
        if query.strip().lower() in ("q", "exit", ""): break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text": print(block.text)
        print()
