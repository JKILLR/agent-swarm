# Life OS: Daily Operations Design

## The Philosophy

Life OS is not a dashboard. It's not an app you open. It's a quiet, intelligent presence that already knows what's going on and tells you what matters when you need to hear it.

J doesn't want to check things. J wants to be told.

---

## How J Talks to Life OS

### Primary Interface: Natural Conversation

J talks to Life OS like talking to a competent assistant who's been following along all day:

```
J: "What's going on?"
Life OS: "Mike texted about the drywall delivery - he's asking if tomorrow works.
         You've got a site meeting at 2pm and three emails from the architect
         about the mechanical room changes. Also, you're behind on two POs -
         the plumbing fixtures and the exterior lighting."

J: "Tell Mike tomorrow's good and draft those POs"
Life OS: "Done - texted Mike. For the POs, I need supplier names.
         The plumbing fixtures quote was from ABC Supply.
         Do you want me to pull that up?"
```

No commands. No syntax. Just conversation.

### When J Asks

| J Says | Life OS Does |
|--------|--------------|
| "What's going on?" | Morning briefing - urgent items first |
| "Who's trying to reach me?" | Summarize unreturned communications |
| "What's on for today?" | Calendar + pending items relevant to today |
| "Handle my emails" | Triage: draft responses, flag important, archive noise |
| "What do I need to do?" | Task list, prioritized by urgency |
| "Text [name] and tell them..." | Compose and send iMessage |
| "Email [name] about..." | Draft email (may send or queue for review) |
| "I need a PO for..." | Start PO workflow (see below) |
| "What's the status on [project/item]?" | Context lookup, recent communications, docs |
| "Generate tasks for tomorrow" | Create laborer task list for site |

---

## When Life OS Reaches Out

### Proactive Alerts (Push to J)

Life OS doesn't wait to be asked when:

**IMMEDIATE** (within minutes):
- Text from a sub that looks like they're waiting on J
- Email marked urgent or with time-sensitive language
- Calendar conflict detected
- Someone trying to reach J through multiple channels

**DAILY DIGEST** (morning):
- Communications needing response (prioritized)
- Today's schedule overview
- Pending items that are aging out
- POs that need attention

**SMART REMINDERS**:
- "You told Mike you'd get back to him about the schedule - that was yesterday"
- "The plumbing PO has been sitting for 4 days"
- "You have a meeting in 30 minutes with the architect"

### Alert Channels

| Urgency | How |
|---------|-----|
| Critical | Text J directly (or designated alert method) |
| Daily digest | Morning message or chat interface |
| Context reminders | Appear when J next engages |

---

## What Runs Silently vs Needs Approval

### Silent (Auto-Execute)

Life OS does these without asking:

- **Read and index** all incoming emails and texts
- **Categorize** communications (work/personal, by project, by contact)
- **Update context** about projects, contacts, and ongoing threads
- **Detect patterns** (e.g., "Mike usually asks about deliveries on Mondays")
- **Calendar analysis** - conflicts, prep time needed
- **Archive spam/noise** after learning J's preferences
- **Pull relevant docs** when a topic comes up

### Needs One-Click Approval

J sees what Life OS drafted, approves or edits:

- **Email drafts** - "I drafted this response. Send?" [Send / Edit / Skip]
- **Meeting accepts** - "Accept this meeting?" [Yes / No / Suggest time]
- **PO submissions** - "Ready to submit this PO?" [Submit / Edit / Hold]

### Needs J to Provide Info

- New supplier details
- Decisions on scheduling conflicts
- Priority calls when multiple urgent items
- Amounts/quantities for POs

---

## The PO Workflow

J hates writing POs. They pile up. Here's how Life OS handles it:

### Trigger Options

1. **J asks**: "I need a PO for the cabinet hardware"
2. **Life OS detects**: Quote email from supplier → suggests PO
3. **Life OS reminds**: "This quote expires in 3 days - want me to draft the PO?"

### Workflow

```
Step 1: Life OS gathers context
        - Searches emails for the quote
        - Pulls supplier info from past POs or contacts
        - Checks project context for job number, delivery address

Step 2: Life OS drafts the PO form data
        - Pre-fills everything it knows
        - Lists what's missing

Step 3: J fills gaps conversationally
        Life OS: "Got the PO ready for ABC Supply - cabinet hardware, $4,200.
                  I need: quantity per item and desired delivery date."
        J: "48 units, delivery Friday next week"

Step 4: Life OS populates Google Form
        - Opens the form (or submits via API if available)
        - Shows J preview: "Here's the PO. Submit?"

Step 5: J approves
        - One-click submit
        - Life OS files copy in Drive, updates project docs
```

### PO Intelligence

Over time, Life OS learns:
- Which suppliers J uses for what
- Typical quantities and pricing
- Lead times for different items
- J's formatting preferences

---

## Laborer Task Generation

### When

- J says "generate tasks for tomorrow" or "what do the guys need to do?"
- Morning digest can include suggested task list
- After site meeting, J says "break that down into tasks"

### How It Works

```
J: "Generate tasks for tomorrow at Langley"

Life OS considers:
- Current project phase (from context)
- Recent instructions from architect emails
- Open items from previous task lists
- Weather forecast (outdoor work?)
- Scheduled inspections

Life OS: "Here's tomorrow's task list for Langley 5:

         Priority:
         1. Finish framing in Unit 204 - mechanical room wall
         2. Install backing for bathroom accessories (Units 201-204)

         Secondary:
         3. Clean up debris from yesterday's drywall delivery
         4. Stage material for Thursday's electrical rough-in

         Note: Electrical inspection is Thursday AM - make sure rough-in is complete by end of day tomorrow.

         Print this / Send to [foreman name] / Edit?"
```

### Task List Outputs

- **Print-ready format** - simple text J can print for the site
- **Send to foreman** - text or email the list to site lead
- **Track completion** - if J wants, check items off through the day

---

## The Morning Briefing

Every morning, Life OS is ready with this when J asks "what's going on?" or checks in:

```
Good morning.

TODAY:
- Site meeting at 2pm with Structural Engineer (Teams link in your calendar)
- Mike's expecting you to confirm delivery window for drywall
- Rain forecast after 4pm

NEEDS YOUR ATTENTION:
1. Architect emailed twice about the mechanical room spec change - sounds like they need direction
2. City permit office called (missed call from 604-XXX-XXXX) - probably about the variance
3. You're 5 days behind on the plumbing fixtures PO

WAITING ON OTHERS:
- Cabinet quote request sent Tuesday - no response yet
- Structural engineer hasn't sent revised drawings

QUIET (no action needed):
- 4 supplier newsletters archived
- Mike confirmed material counts yesterday

What would you like to tackle first?
```

---

## Communication Patterns

### Who Contacts J How

| Contact Type | Channel | Life OS Response |
|--------------|---------|------------------|
| Subs | Text (iMessage) | Read, surface in digest, can respond via text |
| Architects/Engineers | Email | Read, draft responses, organize by project |
| City/Permits | Phone/Email | Flag immediately - these are often time-sensitive |
| Office/Developer | Email | Read, prioritize based on sender |
| Suppliers | Email | Parse quotes, track status, link to PO workflow |

### Response Drafting Intelligence

Life OS learns J's voice:
- Short, direct responses to subs ("Yep that works", "See you at 8")
- Professional but not formal with architects ("Got the changes, reviewing now")
- Formal with city ("Thank you for your call. I'll have the documentation to you by...")

J can adjust any draft before sending.

---

## Integration Details

### Gmail (Connected)
- Read all incoming email
- Send emails (with J approval)
- Search for quotes, documents, specific communications
- Learn J's email style from sent messages

### Google Calendar (Connected)
- Read all events
- Create events (with J approval)
- Detect conflicts
- Set smart reminders

### Google Drive (Connected)
- Read project files
- Search for specs, drawings, quotes
- Upload generated documents (POs, task lists)
- Reference docs in context

### Google Tasks (Connected)
- Create tasks (with J approval or from task generation)
- Track completion
- Integrate with daily digest

### Google Forms (NEEDS SETUP)
- **Required**: Add Forms API scope to OAuth
- **Use**: Submit PO forms programmatically
- **Fallback**: Pre-fill form URL that J clicks to submit

### iMessage Reader (Connected)
- Search messages by contact or content
- Read recent conversations
- Surface text-based communication in digest
- **Note**: Sending iMessages may require additional setup (AppleScript or Shortcuts)

---

## Privacy & Control

### J is Always in Control

- Life OS reads everything but only acts with permission
- Drafts are always shown before sending
- J can say "don't look at [folder/contact/thread]" to exclude areas
- All actions are logged and reversible where possible
- J can ask "what did you do?" to see recent actions

### Data Stays Local

- Context stays in J's system
- No external services beyond the APIs J has connected
- J's communication patterns and preferences stay private

---

## What Life OS DOESN'T Do

- **Make decisions for J** - it presents options, J decides
- **Send without approval** - drafts are always reviewed
- **Overwhelm with information** - filters and prioritizes
- **Replace human judgment** - knows when to say "you should call them"
- **Work without context** - asks clarifying questions when uncertain

---

## The Experience Goal

J checks their phone in the morning. Life OS has already:
- Read overnight emails
- Seen the texts from subs
- Checked the calendar
- Identified what matters

J says: "What's going on?"

And Life OS says exactly what J needs to hear to start the day - not a list of everything, but the things that actually need attention, in order of importance, with the boring stuff already handled or explicitly marked as "nothing needed."

J's response should be: "Thanks. Let's start with the PO."

Not: "Let me check my email."

---

## Evolution

### Phase 1: Aware
- Life OS reads and understands all channels
- Surfaces information on request
- Drafts responses for approval

### Phase 2: Predictive
- Anticipates what J will ask
- Proactively suggests actions
- Learns J's patterns and preferences

### Phase 3: Autonomous
- Handles routine items automatically
- J focuses only on decisions and exceptions
- Trust built through track record

---

*This system is built around one principle: J should never feel like they're operating software. They should feel like they have help.*
