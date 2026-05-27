# uqlm — Claude Code skill

A single-file [Claude Code](https://www.anthropic.com/claude-code) skill that teaches Claude Code to add `uqlm` hallucination detection to your LLM agents — LangGraph, LangChain, AutoGen, CrewAI, or hand-rolled.

Install once, then in any Claude Code session ask things like:

- *"Add hallucination detection to this LangGraph agent."*
- *"Score the responses my agent generates and retry on low confidence."*
- *"Which uqlm scorer should I use for long-form answers?"*

Claude Code will pick the right scorer, wire it into your code correctly, and suggest a starting threshold.

## What's in this directory

| File | Purpose |
|---|---|
| `SKILL.md` | The skill itself. Frontmatter description + scorer cheat sheet, framework recipes, thresholds, pitfalls. |
| `uqlm.md`  | A `/uqlm` slash command for explicit invocation. |
| `README.md` | This file. |

## Install (user-level — every project)

```bash
mkdir -p ~/.claude/skills/uqlm ~/.claude/commands
curl -fsSL https://raw.githubusercontent.com/cvs-health/uqlm/develop/uqlm/integration/claudecode/SKILL.md \
  > ~/.claude/skills/uqlm/SKILL.md
curl -fsSL https://raw.githubusercontent.com/cvs-health/uqlm/develop/uqlm/integration/claudecode/uqlm.md \
  > ~/.claude/commands/uqlm.md
```

> Replace `develop` with `main` once this lands on the default branch.

## Install (project-level — single repo)

From the root of your project:

```bash
mkdir -p .claude/skills/uqlm .claude/commands
curl -fsSL https://raw.githubusercontent.com/cvs-health/uqlm/develop/uqlm/integration/claudecode/SKILL.md \
  > .claude/skills/uqlm/SKILL.md
curl -fsSL https://raw.githubusercontent.com/cvs-health/uqlm/develop/uqlm/integration/claudecode/uqlm.md \
  > .claude/commands/uqlm.md
```

Commit the `.claude/` directory so teammates working in Claude Code get the same skill automatically.

## Install (from a local clone)

```bash
git clone https://github.com/cvs-health/uqlm.git
mkdir -p ~/.claude/skills/uqlm ~/.claude/commands
cp uqlm/uqlm/integration/claudecode/SKILL.md ~/.claude/skills/uqlm/SKILL.md
cp uqlm/uqlm/integration/claudecode/uqlm.md  ~/.claude/commands/uqlm.md
```

## Verify

In a fresh Claude Code session:

```
/help
```

`uqlm` should be listed under skills. Or just ask: *"add uqlm hallucination detection to my agent"* — the skill auto-loads when the description matches.

To verify the library is installed:

```bash
python -c "from uqlm import BlackBoxUQ; print('ok')"
```

## Uninstall

```bash
rm -rf ~/.claude/skills/uqlm ~/.claude/commands/uqlm.md
# or, for project-level
rm -rf .claude/skills/uqlm .claude/commands/uqlm.md
```

## Requirements

- [Claude Code](https://www.anthropic.com/claude-code).
- `pip install uqlm` (the skill teaches Claude Code to write code using uqlm — installing the library is on the user's side).

## How it works

Claude Code auto-loads skills whose `description` frontmatter matches the user's request. The `SKILL.md` here is triggered by phrases like *hallucination detection*, *confidence score*, *BlackBoxUQ*, *UQLMNode*, etc. Once loaded, the skill body acts as a playbook the model follows — scorer cheat sheet, framework-specific wiring snippets, threshold guidance.

The `/uqlm` slash command is an explicit-invocation alternative: type `/uqlm add scoring to graph.py` and Claude Code runs the same recipe deterministically.

No runtime process, no MCP server, no API keys to configure. It's pure markdown that improves how Claude Code writes uqlm code.
