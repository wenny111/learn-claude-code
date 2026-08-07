#!/usr/bin/env python3
# s09：Memory System，为 Agent 提供跨压缩、跨会话的长期记忆
#
# 存储：
#   .memory/MEMORY.md  轻量索引
#   .memory/*.md       带 YAML 元数据的记忆正文
#
# 主流程：加载索引 → 检索并注入相关记忆 → 压缩上下文
#        → 回合结束后提取新记忆 → 定期合并整理
#
# 运行：python s09_memory/code.py
# 依赖：anthropic、python-dotenv、ANTHROPIC_API_KEY

import os, subprocess, json, time, re
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

WORKDIR = Path.cwd()
# 长期记忆写入磁盘，不受对话压缩和程序重启影响
MEMORY_DIR = WORKDIR / ".memory"; MEMORY_DIR.mkdir(exist_ok=True)
# MEMORY.md 只存目录，完整内容放在同级记忆文件中
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]


# ═══════════════════════════════════════════════════════════
#  NEW in s09: Memory System
# ═══════════════════════════════════════════════════════════

# list 保存允许的 4 种记忆类型
MEMORY_TYPES = ["user", "feedback", "project", "reference"]

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    # split 分段：元数据存 dict，正文存 str，最后用 tuple 一起返回
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


# 写入一条带 YAML 元数据的记忆
def write_memory_file(name: str, mem_type: str, description: str, body: str):
    # 每条记忆独立成文件，便于按需读取和替换
    slug = name.lower().replace(" ", "-").replace("/", "-")
    filename = f"{slug}.md"
    filepath = MEMORY_DIR / filename
    filepath.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
    )
    # 写入后立即更新目录，下一轮即可检索
    _rebuild_index()
    return filepath


# 根据全部记忆文件重建 MEMORY.md 索引
def _rebuild_index():
    # 全量扫描 + 排序：用 list 按文件名顺序累积索引行
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        # 索引只放名称和描述，避免 SYSTEM prompt 携带全部正文
        lines.append(f"- [{name}]({f.name}) — {desc}")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")


# 读取每轮都会注入 SYSTEM 的记忆索引
def read_memory_index() -> str:
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""


# 读取一条记忆的完整内容
def read_memory_file(filename: str) -> str | None:
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()


# 读取全部记忆及其元数据
def list_memory_files() -> list[dict]:
    # list[dict]：每个 dict 代表一条结构化记忆
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })
    return result


# 根据最近对话选择相关记忆；LLM 失败时降级为关键词匹配
def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    # files 是 list[dict]，列表下标同时作为 LLM 可返回的记忆编号
    files = list_memory_files()
    if not files:
        return []

    # 反向扫描消息列表，找到 3 条就停止，相当于固定大小的最近窗口
    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(getattr(b, "text", "")) for b in content
                    if getattr(b, "type", None) == "text"
                )
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    recent = " ".join(reversed(recent_texts))[:2000]

    if not recent.strip():
        return []

    # enumerate 建立“编号 → 记忆”映射，LLM 只需返回编号数组
    catalog_lines = []
    for i, f in enumerate(files):
        catalog_lines.append(f"{i}: {f['name']} — {f['description']}")
    catalog = "\n".join(catalog_lines)

    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}"
    )

    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = extract_text(response.content).strip()
        # 正则截取 JSON 数组，再把合法编号映射回文件名
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            selected = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(files):
                    selected.append(files[idx]["filename"])
                    if len(selected) >= max_items:
                        break
            return selected
    except Exception:
        pass

    # 双层匹配：逐条记忆检查任一关键词是否出现，最多取 max_items 条
    keywords = [w.lower() for w in recent.split() if len(w) > 3]
    selected = []
    for f in files:
        text = (f["name"] + " " + f["description"]).lower()
        if any(kw in text for kw in keywords):
            selected.append(f["filename"])
            if len(selected) >= max_items:
                break
    return selected


# 加载相关记忆正文，准备注入本轮上下文
def load_memories(messages: list) -> str:
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""

    # 先用 list 收集文本，最后 join，避免循环中反复拼接字符串
    parts = ["<relevant_memories>"]
    for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


# 每轮结束后，从最近对话中提取新记忆
def extract_memories(messages: list):
    # messages[-10:] 是固定窗口，只处理最近 10 条消息
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(getattr(b, "text", "")) for b in content
                if getattr(b, "type", None) == "text"
            )
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)

    if not dialogue.strip():
        return

    # 把已有记忆交给 LLM，减少重复提取
    existing = list_memory_files()
    existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=800
        )
        text = extract_text(response.content).strip()
        # LLM 返回 JSON 的 list[dict]，每个 dict 是一条候选记忆
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass


CONSOLIDATE_THRESHOLD = 10

# 达到数量阈值后，合并重复记忆并删除过期内容
def consolidate_memories():
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return

    # 批处理算法：一次读取全部记忆，让 LLM 输出去重后的新列表
    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )

    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )

    try:
        response = client.messages.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3000
        )
        text = extract_text(response.content).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())

        # 整体替换：删除旧文件，再遍历新 list[dict] 重建记忆库
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()

        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)

        print(f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m")
    except Exception:
        pass


# SYSTEM 只常驻轻量索引，详细正文按当前问题单独注入
def build_system() -> str:
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        f"You are a coding agent at {WORKDIR}."
        f"{memories_section}\n"
        "Relevant memories are injected below. Respect user preferences from memory.\n"
        "When the user says 'remember' or expresses a clear preference, extract it as a memory."
    )

SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s08 (skeleton): Basic tools
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired: return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines): lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path); file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content); return f"Wrote {len(content)} bytes to {path}"
    except Exception as e: return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text: return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
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

def extract_text(content) -> str:
    if not isinstance(content, list): return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")

# Subagent (simplified from s06-s07)
SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
]
SUB_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}

def spawn_subagent(description: str) -> str:
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]
    for _ in range(30):
        response = client.messages.create(model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use": break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
    result = extract_text(messages[-1]["content"])
    if not result:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result: break
        if not result: result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result


# ═══════════════════════════════════════════════════════════
#  FROM s08 (skeleton): Compaction pipeline
# ═══════════════════════════════════════════════════════════

CONTEXT_LIMIT = 50000; KEEP_RECENT = 3; PERSIST_THRESHOLD = 30000

def estimate_size(msgs): return len(str(msgs))

def _block_type(block):
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

def _message_has_tool_use(msg):
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
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)

def snip_compact(msgs, mx=50):
    if len(msgs) <= mx: return msgs
    head_end, tail_start = 3, len(msgs) - (mx - 3)
    if head_end > 0 and _message_has_tool_use(msgs[head_end - 1]):
        while head_end < len(msgs) and _is_tool_result_message(msgs[head_end]):
            head_end += 1
    if (tail_start > 0 and tail_start < len(msgs)
            and _is_tool_result_message(msgs[tail_start])
            and _message_has_tool_use(msgs[tail_start - 1])):
        tail_start -= 1
    if head_end >= tail_start:
        return msgs
    return msgs[:head_end] + [{"role": "user", "content": f"[snipped {tail_start - head_end} msgs]"}] + msgs[tail_start:]

def collect_tool_results(msgs):
    blocks = []
    for mi, msg in enumerate(msgs):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list): continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result": blocks.append((mi, bi, block))
    return blocks

def micro_compact(msgs):
    tr = collect_tool_results(msgs)
    if len(tr) <= KEEP_RECENT: return msgs
    for _, _, b in tr[:-KEEP_RECENT]:
        if len(b.get("content", "")) > 120: b["content"] = "[Earlier tool result compacted.]"
    return msgs

def persist_large(tid, out):
    if len(out) <= PERSIST_THRESHOLD: return out
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = TOOL_RESULTS_DIR / f"{tid}.txt"
    if not p.exists(): p.write_text(out)
    return f"<persisted-output>\nFull: {p}\nPreview:\n{out[:2000]}\n</persisted-output>"

def tool_result_budget(msgs, mx=200_000):
    last = msgs[-1] if msgs else None
    if not last or last.get("role") != "user" or not isinstance(last.get("content"), list): return msgs
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= mx: return msgs
    for _, block in sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True):
        if total <= mx: break
        c = str(block.get("content", ""))
        if len(c) <= PERSIST_THRESHOLD: continue
        block["content"] = persist_large(block.get("tool_use_id", "?"), c)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return msgs

def write_transcript(msgs):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    p = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with p.open("w") as f:
        for m in msgs: f.write(json.dumps(m, default=str) + "\n")
    return p

def summarize_history(msgs):
    conv = json.dumps(msgs, default=str)[:80000]
    r = client.messages.create(model=MODEL, messages=[{"role": "user", "content":
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. current goal, 2. key findings, 3. files changed, 4. remaining work, 5. user constraints.\n\n" + conv}],
        max_tokens=2000)
    return extract_text(r.content).strip()

def compact_history(msgs):
    write_transcript(msgs)
    summary = summarize_history(msgs)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]

def reactive_compact(msgs):
    write_transcript(msgs)
    tail_start = max(0, len(msgs) - 5)
    if (tail_start > 0 and tail_start < len(msgs)
            and _is_tool_result_message(msgs[tail_start])
            and _message_has_tool_use(msgs[tail_start - 1])):
        tail_start -= 1
    summary = summarize_history(msgs[:tail_start])
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *msgs[tail_start:]]


# ═══════════════════════════════════════════════════════════
#  Tool Definitions (skeleton — fewer tools to focus on memory)
# ═══════════════════════════════════════════════════════════

TOOLS = [
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
    {"name": "task", "description": "Launch a subagent to handle a subtask.",
     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "task": spawn_subagent,
}


# ═══════════════════════════════════════════════════════════
#  agent_loop — s09: inject memories + extract after each turn
# ═══════════════════════════════════════════════════════════

MAX_REACTIVE_RETRIES = 1

def agent_loop(messages: list):
    # messages 是 list[dict] 消息队列，每项包含 role 和 content
    reactive_retries = 0
    # 每个用户回合只检索一次，本轮后续工具调用复用这些记忆
    memories_content = load_memories(messages)
    # 用列表下标记录当前用户消息，稍后把记忆注入这个位置
    memory_turn = len(messages) - 1 if messages and isinstance(messages[-1].get("content"), str) else None
    # 本轮结束前记忆不会更新，所以 SYSTEM 只构建一次
    system = build_system()

    while True:
        # 压缩前保存原始消息，避免从有损摘要中提取记忆
        pre_compress = [m if isinstance(m, dict) else {"role": m.get("role",""),
            "content": str(m.get("content",""))} for m in messages]

        # 切片赋值原地更新 list，外层持有的 history 引用仍然有效
        messages[:] = tool_result_budget(messages)
        messages[:] = snip_compact(messages)
        messages[:] = micro_compact(messages)

        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        try:
            request_messages = messages
            if memories_content and memory_turn is not None and memory_turn < len(messages):
                # 浅拷贝 list，并替换当前消息 dict，不污染真实对话历史
                request_messages = messages.copy()
                request_messages[memory_turn] = {
                    **messages[memory_turn],
                    "content": memories_content + "\n\n" + messages[memory_turn]["content"],
                }
            response = client.messages.create(
                model=MODEL, system=system, messages=request_messages, tools=TOOLS, max_tokens=8000
            )
            reactive_retries = 0
        except Exception as e:
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue
            raise

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            # 工具链结束后，再提取新记忆并整理旧记忆
            extract_memories(pre_compress)
            consolidate_memories()
            return

        # 工具结果用 list[dict] 收集，再作为一条 user 消息追加
        results = []
        for block in response.content:
            if block.type != "tool_use": continue
            print(f"\033[36m> {block.name}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s09: Memory — persistent cross-session knowledge")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try: query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt): break
        if query.strip().lower() in ("q", "exit", ""): break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text": print(block.text)
        print()
