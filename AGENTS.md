# Agent Instructions

## Terminal Usage

### RTK Wrapper for Noisy Commands

Minimize terminal output to save context window tokens by wrapping noisy commands in the `rtk` wrapper. Don't rely on judgment to skip it — follow this decision framework:

1. **Build & compile (strictly required):** any command that builds or links code (`cmake`, `ninja`, `make`, `clang`) -> `rtk <command>`
2. **Testing (strictly required):** any command that runs test suites (`ctest`, `lit`, `./build/bin/test`) -> `rtk <command>`
3. **Search & IO (strictly required):** any command that reads files or searches directories (`grep`, `find`, `ls`, `cat`, `git status`) -> `rtk <command>`
4. **Silent state changes (exempt):** purely silent operations that only change environment state (`cd`, `mkdir`, `rm`, `export`) -> `rtk` not needed

Mental model: if the command produces more than 5 lines of stdout/stderr, wrap it in `rtk`.

Example: `ninja -C build` (bad) vs. `rtk ninja -C build` (good)

### Autonomy & Safety

- **Auto-run without asking:** compile/build, test, `git status`/`diff`/`log`, install deps, run binaries, read-only queries (`grep`, `ls`, `cat`).
- **Never auto-run — ask first:** `rm -rf`, bulk deletes, `git clean -fdx`, `mkfs`/`dd`/disk format, `git reset --hard`, force-push to main, broad `sudo` destructive ops.

## MCP Tools: code-review-graph

**IMPORTANT: This project has a structural knowledge graph. ALWAYS use the `code-review-graph` MCP tools BEFORE using generic Grep/Glob/Read to explore the codebase.** The graph is faster, cheaper (fewer tokens), and gives structural context (callers, dependents, test coverage) that raw file scanning cannot. Fall back to Grep/Glob/Read only when the graph doesn't cover what you need.

### When to use graph tools first

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with `callers_of`/`callees_of`/`imports_of`/`tests_for`
- **Architecture questions**: `get_architecture_overview` + `list_communities`

### Key tools

| Tool | Use when |
| --- | --- |
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` with `pattern="tests_for"` to check test coverage.

### Subagent MCP policy

All Task / audit / implementer subagents may use any enabled MCP tool without asking the user first. Prefer `code-review-graph` for exploration.
