# Life OS v1 - Minimal Viable Scope

## Philosophy
**Ship fast, learn, extend.** Get message search + basic contacts working in days, not weeks.

---

## V1 Scope (The 20% That Gives 80% Value)

### Core Features

| Feature | Description | Why V1 |
|---------|-------------|--------|
| **Message Search** | Semantic search across iMessage history | Primary use case |
| **Contact Context** | Basic name/phone mapping for messages | Makes search useful |
| **Simple API** | `/search?q=...` endpoint | Easy integration |

### What V1 Does NOT Include (Deferred to V2+)

| Feature | Why Defer |
|---------|-----------|
| Calendar integration | Separate data source, adds complexity |
| Relationship scoring | Nice-to-have, not core |
| Meeting prep automation | Requires calendar + more context |
| Message drafting | Needs style analysis, more training |
| Real-time sync | Batch refresh is fine for v1 |
| Full encryption | Local-only for now |

---

## V1 Architecture

```
┌──────────────────────────────────────────────────────┐
│                    LIFE OS V1                        │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐    ┌─────────────┐                 │
│  │  iMessage   │───▶│  Indexer    │                 │
│  │  chat.db    │    │  (batch)    │                 │
│  └─────────────┘    └──────┬──────┘                 │
│                            │                        │
│  ┌─────────────┐           ▼                        │
│  │  Contacts   │    ┌─────────────┐                 │
│  │AddressBook.db│──▶│  FAISS      │                 │
│  └─────────────┘    │  Index      │                 │
│                     └──────┬──────┘                 │
│                            │                        │
│                            ▼                        │
│                     ┌─────────────┐                 │
│                     │  REST API   │                 │
│                     │  /search    │                 │
│                     └─────────────┘                 │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Data Access (Day 1)
- [ ] Read iMessage chat.db (SQLite)
- [ ] Read AddressBook.db for contact names
- [ ] Map phone numbers → names

### Phase 2: Indexing (Day 1-2)
- [ ] Embed messages with MiniLM (existing embedding service)
- [ ] Store in FAISS index
- [ ] Include metadata: sender, timestamp, contact name

### Phase 3: Search API (Day 2)
- [ ] `/api/life/search?q=...` endpoint
- [ ] Return top N results with context
- [ ] Basic filters: date range, contact

### Phase 4: Integration (Day 3)
- [ ] Hook into COO context (optional)
- [ ] Simple CLI for testing

---

## Technical Details

### Database Locations
```
iMessage: ~/Library/Messages/chat.db
Contacts: ~/Library/Application Support/AddressBook/AddressBook-v22.abcddb
```

### Required Permissions
- Full Disk Access (for chat.db) - Terminal needs this

### Data Model (V1)

```python
@dataclass
class IndexedMessage:
    id: str                    # Message ROWID
    text: str                  # Message content
    sender: str                # Phone/email
    sender_name: str | None    # Resolved contact name
    timestamp: datetime
    is_from_me: bool
    embedding: list[float]     # 384-dim vector
```

### Memory Budget (8GB Mac Mini)
- MiniLM model: ~90MB
- FAISS index (50K messages): ~80MB
- Contact cache: ~5MB
- **Total: <200MB** ✓

---

## API Design (V1)

### Search Endpoint
```
GET /api/life/search?q=<query>&limit=10&contact=<name>&days=30
```

Response:
```json
{
  "query": "dinner plans",
  "results": [
    {
      "text": "Want to grab dinner Thursday?",
      "sender_name": "Sarah",
      "timestamp": "2025-01-03T18:30:00",
      "is_from_me": false,
      "score": 0.87
    }
  ],
  "total_indexed": 48392
}
```

### Reindex Endpoint
```
POST /api/life/reindex
```
Triggers batch re-sync of messages (runs in background).

---

## Security (V1 - Minimal)

| Concern | V1 Approach |
|---------|-------------|
| API Access | localhost only (127.0.0.1 binding) |
| Data Storage | Index stored locally, same permissions as chat.db |
| Sensitive Content | Basic filter (SSN/CC patterns) - skip indexing |

**Deferred**: Encryption at rest, API auth, audit logging

---

## Upgrade Path to V2

V1 is designed so V2 features are **additive, not rewrites**:

| V2 Feature | How V1 Enables It |
|------------|-------------------|
| Calendar | New data source → same indexing pipeline |
| Relationship graph | Contact cache → add edge metadata |
| Meeting prep | Search API + Calendar → composition |
| Real-time sync | Batch indexer → add file watcher |
| Encryption | FAISS index → wrap with encryption layer |

---

## Success Criteria

V1 is done when:
1. ✅ Can search "what did [person] say about [topic]"
2. ✅ Returns results in <500ms
3. ✅ Indexes 50K+ messages without OOM
4. ✅ Contact names resolve correctly
5. ✅ API accessible from agent-swarm backend

---

## Files to Create

```
backend/
├── services/
│   └── life_os/
│       ├── __init__.py
│       ├── message_reader.py    # Read chat.db
│       ├── contact_reader.py    # Read AddressBook
│       └── message_index.py     # FAISS indexing
└── routes/
    └── life.py                  # /api/life/* endpoints
```

---

## Questions Before Starting

1. **Do you have Full Disk Access enabled for Terminal?** (System Preferences → Privacy → Full Disk Access)
2. **Approximate message volume?** (Affects initial index time)
3. **Want a simple CLI test first, or go straight to API?**
