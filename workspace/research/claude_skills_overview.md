# Claude Code Skills - Complete Overview

## What ARE Skills?

Skills are **folders of instructions, scripts, and resources that Claude loads dynamically** to improve performance on specialized tasks. They are modular, self-contained packages that extend Claude's capabilities with:

- Specialized workflows
- Tool integrations
- Domain expertise
- Bundled resources (scripts, reference docs, templates)

**Key characteristic**: Skills are triggered **automatically by Claude based on your request**, not manually. Claude decides when to use them based on the skill's description and the task context. This is the fundamental difference from slash commands.

> "Skills aren't separate processes, sub-agents, or external tools: they're injected instructions that guide Claude's behavior within the main conversation."

---

## Skills vs Slash Commands

| Feature | Skills | Slash Commands |
|---------|--------|----------------|
| **Invocation** | Automatic (Claude decides) | Manual (`/command-name`) |
| **Complexity** | Rich workflows with supporting files | Single-file prompts |
| **Structure** | Directory with SKILL.md + resources | Single .md file |
| **Discovery** | Claude auto-applies based on context | User types command |
| **Use case** | Complex, conditional knowledge | Quick, repetitive tasks |

**When to use Skills:**
- Complex, multi-step workflows
- Claude should auto-apply knowledge when relevant
- Need supporting files (scripts, references, templates)

**When to use Slash Commands:**
- Quick, atomic actions
- Want explicit control over when it runs
- Simple, frequently used operations

Both can coexist in a project.

---

## File Format & Structure

### Required Structure

A skill is a **directory** containing at minimum a `SKILL.md` file:

```
my-skill/
├── SKILL.md           # Required - Entry point with frontmatter + instructions
├── scripts/           # Optional - Executable Python/Bash scripts
├── references/        # Optional - Documentation loaded into context
└── assets/            # Optional - Templates, icons, files for output
```

### SKILL.md Format

The SKILL.md file has two parts:

1. **YAML Frontmatter** (between `---` markers) - Metadata
2. **Markdown Body** - Instructions for Claude

#### Required Frontmatter Fields

```yaml
---
name: my-skill-name
description: A clear description of what this skill does and when to use it
---
```

| Field | Requirements |
|-------|--------------|
| `name` | Max 64 chars, lowercase letters/numbers/hyphens only, no reserved words (anthropic, claude) |
| `description` | Max 1024 chars, non-empty, describes WHAT it does AND WHEN to use it |

#### Optional Frontmatter Fields

```yaml
---
name: my-skill-name
description: Description here
version: 1.0.0
allowed-tools: ["Read", "Write", "Bash"]  # Limit available tools
---
```

### Complete Example

```yaml
---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
---

# PDF Processing

## Quick Start

Extract text with pdfplumber:

```python
import pdfplumber

with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```

## Advanced Features

- **Form filling**: See [FORMS.md](FORMS.md) for complete guide
- **API reference**: See [REFERENCE.md](REFERENCE.md) for all methods

## Workflow

1. Analyze the PDF structure
2. Extract text or fill fields as needed
3. Validate output
```

---

## How Skills Work (Progressive Disclosure)

Skills use a **three-level loading system** to optimize context window usage:

### Level 1: Metadata (Always Loaded)
At startup, only `name` and `description` from all skills are pre-loaded into the system prompt (~100 words per skill). This lets Claude decide relevance without loading full content.

### Level 2: SKILL.md Body (On Trigger)
When Claude determines the skill is relevant, it reads the full SKILL.md file (keep under 500 lines / ~5k words).

### Level 3: Bundled Resources (As Needed)
Additional files (references, scripts, assets) are loaded only when specific tasks require them.

> "The context window is a public good. Your Skill shares it with everything else Claude needs to know."

---

## Skill Locations

### User Skills (Personal)
```
~/.claude/skills/
└── my-skill/
    └── SKILL.md
```

### Project Skills (Via Plugins)
```
my-plugin/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── my-skill/
        └── SKILL.md
```

### Installing from Anthropic's Repository
```bash
# Register the marketplace
/plugin marketplace add anthropics/skills

# Install specific skills
/plugin install document-skills@anthropic-agent-skills
/plugin install example-skills@anthropic-agent-skills
```

---

## Best Practices

### Writing Effective Descriptions

The description is **critical** - Claude uses it to decide when to invoke your skill.

**Good descriptions:**
```yaml
description: Extract text and tables from PDF files, fill forms, merge documents. Use when working with PDF files or when the user mentions PDFs, forms, or document extraction.
```

**Bad descriptions:**
```yaml
description: Helps with documents  # Too vague
description: Does stuff with files  # No context for when to use
```

**Rules:**
- Write in third person ("Processes files" not "I can help you")
- Include specific trigger terms
- Describe both WHAT and WHEN

### Keep SKILL.md Concise

- Under 500 lines for optimal performance
- Only add what Claude doesn't already know
- Use references for detailed content
- Claude is already smart - don't over-explain

**Good (concise):**
```markdown
## Extract PDF text

Use pdfplumber:
```python
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = pdf.pages[0].extract_text()
```
```

**Bad (verbose):**
```markdown
## Extract PDF text

PDF files are a common file format that contains text and images.
To extract text you need a library. There are many libraries but
we recommend pdfplumber because it's easy to use...
```

### Use Progressive Disclosure Patterns

**Pattern 1: High-level guide with references**
```markdown
# PDF Processing

## Quick start
[Basic examples here]

## Advanced features
- **Form filling**: See [FORMS.md](FORMS.md)
- **API reference**: See [REFERENCE.md](REFERENCE.md)
```

**Pattern 2: Domain-specific organization**
```
bigquery-skill/
├── SKILL.md (overview and navigation)
└── references/
    ├── finance.md
    ├── sales.md
    └── product.md
```

### Keep References One Level Deep

Claude may partially read deeply nested files.

**Bad:**
```markdown
# SKILL.md → advanced.md → details.md → actual info
```

**Good:**
```markdown
# SKILL.md
- Basic: [in SKILL.md]
- Advanced: See [advanced.md]
- API: See [reference.md]
```

### Bundle Utility Scripts

Scripts run without loading into context - only output consumes tokens.

```markdown
## Utility scripts

**analyze_form.py**: Extract form fields
```bash
python scripts/analyze_form.py input.pdf > fields.json
```

**validate.py**: Check for errors
```bash
python scripts/validate.py fields.json
```
```

---

## Naming Conventions

Use **gerund form** (verb + -ing) for skill names:

**Good:**
- `processing-pdfs`
- `analyzing-spreadsheets`
- `managing-databases`
- `testing-code`

**Avoid:**
- `helper`, `utils`, `tools` (vague)
- `anthropic-helper` (reserved words)

---

## Built-in Skills from Anthropic

Anthropic provides production-grade document skills:

- **DOCX** - Word documents with tracked changes, comments, formatting
- **PPTX** - PowerPoint with layouts, templates, charts
- **XLSX** - Excel with formulas, data analysis
- **PDF** - PDF reading and form filling

Install via:
```bash
/plugin install document-skills@anthropic-agent-skills
```

---

## Skill Creation Workflow

### Step 1: Understand the Use Case
Run Claude on tasks without a skill first. Document what information you repeatedly provide.

### Step 2: Plan Reusable Content
Identify what should become:
- Scripts (repeated code operations)
- References (documentation, schemas)
- Assets (templates, boilerplate)

### Step 3: Create the Skill

```bash
# Using skill-creator (if installed)
python scripts/init_skill.py my-skill --path ./skills

# Or manually create:
mkdir -p ~/.claude/skills/my-skill
touch ~/.claude/skills/my-skill/SKILL.md
```

### Step 4: Write SKILL.md
- Add required frontmatter
- Write concise instructions
- Add reference files for detailed content
- Add scripts for deterministic operations

### Step 5: Test and Iterate
- Test with real tasks
- Observe how Claude uses the skill
- Refine based on behavior

---

## Checklist for Effective Skills

### Core Quality
- [ ] Description is specific with trigger terms
- [ ] Description includes WHAT and WHEN
- [ ] SKILL.md under 500 lines
- [ ] Detailed content in separate files
- [ ] Consistent terminology
- [ ] Concrete examples
- [ ] References one level deep

### Scripts (if applicable)
- [ ] Scripts handle errors explicitly
- [ ] Required packages listed
- [ ] No Windows-style paths (use `/`)
- [ ] Validation steps for critical operations

### Testing
- [ ] Tested with real scenarios
- [ ] Works with intended model (Haiku/Sonnet/Opus)

---

## Sources

### Official Anthropic Documentation
- [Agent Skills - Claude Code Docs](https://code.claude.com/docs/en/skills)
- [How to Create Custom Skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills)
- [Skill Authoring Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- [Equipping Agents with Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Introducing Agent Skills](https://www.anthropic.com/news/skills)

### Official Repository
- [GitHub - anthropics/skills](https://github.com/anthropics/skills)

### Community Resources
- [Inside Claude Code Skills: Structure, Prompts, Invocation](https://mikhail.io/2025/10/claude-code-skills/)
- [Claude Skills Tutorial](https://www.siddharthbharath.com/claude-skills/)
- [Claude Agent Skills: Deep Dive](https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/)
