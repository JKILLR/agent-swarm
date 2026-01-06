# Life OS Architecture

## Personal Context Engine for Integrated Life Intelligence

---

## 1. Executive Summary

### Vision

Life OS is an **external brain interface** that provides semantic access to your entire life context. By integrating iOS personal data (Messages, Contacts, Calendar) with project context through a unified embedding layer, it enables natural language queries across all personal information.

### Key Capabilities

- **Semantic Message Search**: "What did Sarah say about the Denver trip?" instantly retrieves relevant conversations
- **Relationship Intelligence**: Understands communication patterns, relationship strength, and context history per person
- **Meeting Prep**: Auto-generates briefings with relevant history, recent conversations, and context before any meeting
- **Draft Writing**: Generates contextually-aware message drafts matching your communication style per recipient
- **Temporal Awareness**: Correlates events, messages, and projects across time for comprehensive context

### Design Principles

1. **Privacy First**: All data stays local on your Mac Mini
2. **Memory Efficient**: Designed for 8GB RAM constraint
3. **Lazy Loading**: Only load what's needed, when needed
4. **Incremental Sync**: Process only new/changed data

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LIFE OS ENGINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────────┐ │
│  │  TIER 1     │    │    TIER 2       │    │         TIER 3               │ │
│  │  IDENTITY   │    │  PEOPLE GRAPH   │    │        TEMPORAL              │ │
│  │             │    │                 │    │                              │ │
│  │ ┌─────────┐ │    │ ┌─────────────┐ │    │ ┌──────────┐ ┌────────────┐ │ │
│  │ │profile  │ │    │ │  Contacts   │ │    │ │ Messages │ │  Calendar  │ │ │
│  │ │.yaml    │ │    │ │  (500+)     │ │    │ │ (50K+)   │ │  Events    │ │ │
│  │ └─────────┘ │    │ └─────────────┘ │    │ └──────────┘ └────────────┘ │ │
│  │ ┌─────────┐ │    │ ┌─────────────┐ │    │ ┌──────────┐ ┌────────────┐ │ │
│  │ │comm_    │ │    │ │Relationship │ │    │ │ Projects │ │   Notes    │ │ │
│  │ │style    │ │    │ │  Scores     │ │    │ │          │ │            │ │ │
│  │ └─────────┘ │    │ └─────────────┘ │    │ └──────────┘ └────────────┘ │ │
│  │ ┌─────────┐ │    │ ┌─────────────┐ │    │                              │ │
│  │ │prefs    │ │    │ │  Context    │ │    │      [EMBEDDED VECTORS]      │ │
│  │ │.yaml    │ │    │ │  History    │ │    │      384-dim MiniLM-L6       │ │
│  │ └─────────┘ │    │ └─────────────┘ │    │                              │ │
│  │             │    │                 │    │                              │ │
│  │ ~500 tokens │    │   Lazy Load     │    │     Semantic Search          │ │
│  │ Always Hot  │    │   Per-Person    │    │     via Embeddings           │ │
│  └─────────────┘    └─────────────────┘    └──────────────────────────────┘ │
│         │                   │                          │                     │
│         └───────────────────┼──────────────────────────┘                     │
│                             ▼                                                │
│              ┌──────────────────────────────┐                                │
│              │   UNIFIED CONTEXT ENGINE     │                                │
│              │                              │                                │
│              │  • Context Assembly          │                                │
│              │  • Priority Ranking          │                                │
│              │  • Token Budget Management   │                                │
│              └──────────────────────────────┘                                │
│                             │                                                │
│                             ▼                                                │
│              ┌──────────────────────────────┐                                │
│              │       QUERY ROUTER           │                                │
│              │                              │                                │
│              │  • Intent Classification     │                                │
│              │  • Source Selection          │                                │
│              │  • Result Fusion             │                                │
│              └──────────────────────────────┘                                │
│                             │                                                │
├─────────────────────────────┼────────────────────────────────────────────────┤
│                             ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         iOS DATA LAYER                                 │ │
│  │                                                                        │ │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐    │ │
│  │   │   chat.db    │    │ AddressBook  │    │   Calendar.sqlitedb  │    │ │
│  │   │  (iMessage)  │    │   (SQLite)   │    │    (EventKit API)    │    │ │
│  │   └──────────────┘    └──────────────┘    └──────────────────────┘    │ │
│  │                                                                        │ │
│  │   ~/Library/Messages  ~/Library/App Support  ~/Library/Calendars      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Query: "What did John say about the project timeline?"
                    │
                    ▼
           ┌───────────────┐
           │ Query Router  │
           │               │
           │ Intent: SEARCH│
           │ Entity: John  │
           │ Topic: project│
           └───────┬───────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │Identity │ │ People  │ │Temporal │
   │ Context │ │  Graph  │ │ Search  │
   └────┬────┘ └────┬────┘ └────┬────┘
        │          │          │
        │    Load John's      │
        │    relationship     │
        │    context     Embed query,
        │          │     search messages
        │          │          │
        └──────────┼──────────┘
                   ▼
           ┌───────────────┐
           │   Context     │
           │   Assembly    │
           │               │
           │ Rank + Merge  │
           │ Token Budget  │
           └───────────────┘
                   │
                   ▼
            Final Response
```

---

## 3. Three-Tier Memory Model

### TIER 1: Identity Layer (Always Loaded)

The Identity Layer contains core information about you that shapes all interactions. This is always in memory (~500 tokens, ~50KB).

```yaml
# memory/lifeos/identity/profile.yaml
name: "Jake Ellingson"
location: "Denver, CO"
timezone: "America/Denver"
occupation: "Software Engineer / Entrepreneur"
companies:
  - name: "Current Company"
    role: "Founder"
key_projects:
  - "Agent Swarm"
  - "Life OS"
communication_principles:
  - "Direct and concise"
  - "Technical depth when relevant"
  - "Friendly but professional"
```

```yaml
# memory/lifeos/identity/communication_style.yaml
default_tone: "professional-casual"
email_signature: true
response_length: "concise"
formality_by_context:
  work: "professional"
  friends: "casual"
  family: "warm"
patterns:
  greeting: "Hey {name},"
  signoff_casual: "- Jake"
  signoff_formal: "Best, Jake"
```

```yaml
# memory/lifeos/identity/preferences.yaml
working_hours:
  start: "08:00"
  end: "18:00"
  timezone: "America/Denver"
meeting_preferences:
  buffer_minutes: 15
  max_per_day: 5
  preferred_times: ["10:00", "14:00", "16:00"]
notification_quiet_hours:
  start: "21:00"
  end: "08:00"
priorities:
  - "Deep work blocks"
  - "Family time"
  - "Health/Exercise"
```

### TIER 2: People Graph (Lazy Loaded)

The People Graph maintains relationship context for every person you interact with. Loaded on-demand per person.

```python
# Data structure per person
class PersonNode:
    # Core identity (from Contacts)
    id: str                      # Unique identifier
    name: str                    # Full name
    phone_numbers: List[str]     # For message matching
    emails: List[str]            # For email matching

    # Relationship metadata
    relationship_type: str       # "work", "friend", "family", "acquaintance"
    relationship_score: float    # 0.0 - 1.0 based on interaction frequency
    first_contact: datetime      # When you first connected
    last_contact: datetime       # Most recent interaction

    # Communication patterns
    preferred_channel: str       # "imessage", "email", "slack"
    avg_response_time: float     # Their typical response time (hours)
    communication_style: str     # "formal", "casual", "technical"

    # Context
    shared_projects: List[str]   # Projects you work on together
    topics_discussed: List[str]  # Common conversation topics
    important_dates: Dict        # Birthdays, anniversaries
    notes: str                   # Manual notes about this person
```

**Relationship Scoring Algorithm:**

```python
def calculate_relationship_score(person_id: str) -> float:
    """
    Score from 0.0 to 1.0 based on:
    - Message frequency (40%)
    - Recency of contact (30%)
    - Meeting frequency (20%)
    - Manual boost (10%)
    """
    messages_30d = count_messages(person_id, days=30)
    messages_score = min(messages_30d / 50, 1.0) * 0.4

    days_since_contact = (now() - last_contact(person_id)).days
    recency_score = max(0, 1 - days_since_contact / 90) * 0.3

    meetings_30d = count_meetings(person_id, days=30)
    meeting_score = min(meetings_30d / 10, 1.0) * 0.2

    manual_boost = get_manual_boost(person_id) * 0.1

    return messages_score + recency_score + meeting_score + manual_boost
```

### TIER 3: Temporal Layer (Embedded & Searchable)

The Temporal Layer contains all time-based data, embedded for semantic search.

**Message Storage:**

```python
class MessageChunk:
    id: str
    conversation_id: str         # Groups messages in same thread
    participants: List[str]      # Person IDs involved
    timestamp: datetime
    content: str                 # The actual message text
    embedding: np.ndarray        # 384-dim vector from MiniLM-L6

    # Metadata for filtering
    is_from_me: bool
    channel: str                 # "imessage", "email", etc.
    has_attachments: bool
```

**Conversation Chunking Strategy:**

```python
def chunk_conversation(messages: List[Message]) -> List[MessageChunk]:
    """
    Group messages into semantic chunks for embedding.
    Rules:
    1. Same conversation thread
    2. Within 4-hour window
    3. Max 500 tokens per chunk
    4. Preserve context with overlap
    """
    chunks = []
    current_chunk = []
    current_tokens = 0
    last_time = None

    for msg in messages:
        time_gap = (msg.timestamp - last_time).hours if last_time else 0
        msg_tokens = count_tokens(msg.content)

        if time_gap > 4 or current_tokens + msg_tokens > 500:
            if current_chunk:
                chunks.append(create_chunk(current_chunk))
            current_chunk = [msg]
            current_tokens = msg_tokens
        else:
            current_chunk.append(msg)
            current_tokens += msg_tokens

        last_time = msg.timestamp

    return chunks
```

**Calendar Event Storage:**

```python
class CalendarEvent:
    id: str
    title: str
    start_time: datetime
    end_time: datetime
    location: Optional[str]
    attendees: List[str]         # Person IDs
    description: Optional[str]
    embedding: np.ndarray        # Embedded title + description

    # Generated context
    prep_notes: Optional[str]    # Auto-generated meeting prep
    related_messages: List[str]  # Message IDs with relevant context
```

---

## 4. iOS Data Access (macOS)

Life OS accesses iOS data that syncs to macOS. This requires **Full Disk Access** permission for the application.

### Data Sources

| Data Type | Location | Access Method | Sync Frequency |
|-----------|----------|---------------|----------------|
| iMessage | `~/Library/Messages/chat.db` | SQLite (read-only) | Real-time |
| Contacts | `~/Library/Application Support/AddressBook/` | SQLite | On change |
| Calendar | `~/Library/Calendars/` | EventKit API | Real-time |
| Notes | `~/Library/Group Containers/group.com.apple.notes/` | SQLite | On change |

### iMessage Access (chat.db)

```python
# backend/services/ios/message_reader.py

import sqlite3
from pathlib import Path
from datetime import datetime

class MessageReader:
    DB_PATH = Path.home() / "Library/Messages/chat.db"

    def __init__(self):
        self.conn = sqlite3.connect(
            f"file:{self.DB_PATH}?mode=ro",  # Read-only
            uri=True
        )

    def get_messages_since(self, since: datetime) -> List[dict]:
        """Fetch messages since timestamp for incremental sync."""
        # Apple stores dates as nanoseconds since 2001-01-01
        apple_epoch = datetime(2001, 1, 1)
        ns_since = int((since - apple_epoch).total_seconds() * 1e9)

        query = """
        SELECT
            m.ROWID,
            m.guid,
            m.text,
            m.date,
            m.is_from_me,
            m.cache_has_attachments,
            h.id as handle_id,
            c.chat_identifier
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        LEFT JOIN chat c ON cmj.chat_id = c.ROWID
        WHERE m.date > ?
        ORDER BY m.date ASC
        """

        cursor = self.conn.execute(query, (ns_since,))
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_conversation(self, handle_id: str, limit: int = 100) -> List[dict]:
        """Get recent messages with a specific contact."""
        query = """
        SELECT m.*, h.id as handle_id
        FROM message m
        JOIN handle h ON m.handle_id = h.ROWID
        WHERE h.id = ?
        ORDER BY m.date DESC
        LIMIT ?
        """
        cursor = self.conn.execute(query, (handle_id, limit))
        return [self._row_to_dict(row) for row in cursor.fetchall()]
```

### Contacts Access (AddressBook)

```python
# backend/services/ios/contact_reader.py

import sqlite3
from pathlib import Path

class ContactReader:
    DB_PATH = Path.home() / "Library/Application Support/AddressBook/Sources"

    def get_all_contacts(self) -> List[dict]:
        """Read all contacts from AddressBook database."""
        contacts = []

        # Find all AddressBook source databases
        for db_file in self.DB_PATH.glob("*/AddressBook-v22.abcddb"):
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)

            query = """
            SELECT
                r.ZUNIQUEID as id,
                r.ZFIRSTNAME as first_name,
                r.ZLASTNAME as last_name,
                r.ZORGANIZATION as organization,
                r.ZJOBTITLE as job_title,
                r.ZNOTE as notes
            FROM ZABCDRECORD r
            WHERE r.ZFIRSTNAME IS NOT NULL OR r.ZLASTNAME IS NOT NULL
            """

            for row in conn.execute(query):
                contact = self._row_to_contact(row)
                contact['phone_numbers'] = self._get_phone_numbers(conn, row[0])
                contact['emails'] = self._get_emails(conn, row[0])
                contacts.append(contact)

        return contacts

    def _get_phone_numbers(self, conn, record_id: str) -> List[str]:
        """Get all phone numbers for a contact."""
        query = """
        SELECT ZFULLNUMBER FROM ZABCDPHONENUMBER
        WHERE ZOWNER = (SELECT Z_PK FROM ZABCDRECORD WHERE ZUNIQUEID = ?)
        """
        return [row[0] for row in conn.execute(query, (record_id,))]
```

### Calendar Access (EventKit)

```python
# backend/services/ios/calendar_reader.py

import subprocess
import json
from datetime import datetime, timedelta

class CalendarReader:
    """
    Access Calendar via EventKit through a Swift helper.
    EventKit requires proper entitlements and user permission.
    """

    def get_events(self, start: datetime, end: datetime) -> List[dict]:
        """Get calendar events in date range."""
        # Use Swift helper for EventKit access
        result = subprocess.run(
            [
                "swift",
                "scripts/calendar_helper.swift",
                start.isoformat(),
                end.isoformat()
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Calendar access failed: {result.stderr}")

        return json.loads(result.stdout)

    def get_upcoming_meetings(self, hours: int = 24) -> List[dict]:
        """Get meetings in the next N hours."""
        now = datetime.now()
        end = now + timedelta(hours=hours)

        events = self.get_events(now, end)

        # Filter to only meetings (events with attendees)
        return [e for e in events if e.get('attendees')]
```

### Required Permissions

```xml
<!-- Info.plist additions for Full Disk Access -->
<key>NSAppleEventsUsageDescription</key>
<string>Life OS needs access to automate calendar and contacts.</string>

<key>NSCalendarsUsageDescription</key>
<string>Life OS needs calendar access for meeting preparation.</string>

<key>NSContactsUsageDescription</key>
<string>Life OS needs contacts access for relationship intelligence.</string>
```

**Setup Instructions:**

1. System Preferences > Security & Privacy > Privacy > Full Disk Access
2. Add Terminal (or the Life OS application)
3. Grant Calendar and Contacts permissions when prompted

---

## 5. Memory Budget for 8GB Mac Mini

Life OS is designed to operate within strict memory constraints, leaving headroom for other applications and the operating system.

### Memory Allocation

| Component | Max RAM | Notes |
|-----------|---------|-------|
| Embedding Model (MiniLM-L6) | 500MB | Loaded once, shared |
| Identity Context | 50KB | Always in memory |
| People Graph (active subset) | 5MB | ~100 active contacts |
| Message Embeddings Cache | 50MB | ~130K vectors cached |
| Calendar Events Cache | 5MB | 1 year of events |
| Query Processing Buffer | 10MB | Temporary allocations |
| SQLite Connections | 20MB | Read-only, pooled |
| **Total Life OS** | **<600MB** | Target budget |

### Memory Management Strategies

```python
# backend/services/lifeos/memory_manager.py

class LifeOSMemoryManager:
    MAX_CACHED_PEOPLE = 100      # LRU cache for people graph
    MAX_CACHED_EMBEDDINGS = 50   # MB for embedding cache
    EMBEDDING_BATCH_SIZE = 100   # Process in batches

    def __init__(self):
        self.people_cache = LRUCache(maxsize=self.MAX_CACHED_PEOPLE)
        self.embedding_cache = DiskBackedCache(
            max_memory_mb=self.MAX_CACHED_EMBEDDINGS,
            disk_path="data/lifeos/embedding_cache"
        )

    def get_person_context(self, person_id: str) -> PersonNode:
        """Lazy load person context with LRU caching."""
        if person_id in self.people_cache:
            return self.people_cache[person_id]

        # Load from disk
        person = self._load_person_from_db(person_id)
        self.people_cache[person_id] = person
        return person

    def search_messages(self, query_embedding: np.ndarray, k: int = 10):
        """
        Search with disk-backed index.
        Only load embeddings needed for search.
        """
        # Use FAISS with memory-mapped index
        return self.embedding_index.search(query_embedding, k)
```

### Embedding Storage Strategy

```python
# Store embeddings in memory-mapped files for efficient access

class EmbeddingStore:
    """
    Memory-mapped embedding storage.
    384-dim float32 = 1.5KB per embedding
    50MB cache = ~33,000 embeddings in memory
    Full index on disk, memory-mapped for search
    """

    def __init__(self, path: str):
        self.index_path = Path(path) / "embeddings.index"
        self.metadata_path = Path(path) / "metadata.db"

        # FAISS index with memory mapping
        self.index = faiss.read_index(
            str(self.index_path),
            faiss.IO_FLAG_MMAP  # Memory-map instead of loading
        )

    def add_embeddings(self, embeddings: np.ndarray, metadata: List[dict]):
        """Add new embeddings in batches to manage memory."""
        batch_size = 1000

        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            self.index.add(batch)
            self._save_metadata(metadata[i:i+batch_size])

            # Force garbage collection after large batches
            if i % 10000 == 0:
                gc.collect()
```

---

## 6. Implementation Phases

### Phase 1: iOS Sync Layer (Foundation)

**Goal:** Establish reliable data extraction from iOS sources.

**Components:**
- [ ] `MessageReader` - iMessage extraction from chat.db
- [ ] `ContactReader` - Contacts extraction from AddressBook
- [ ] `CalendarReader` - Calendar events via EventKit
- [ ] `SyncManager` - Orchestrates incremental syncing
- [ ] `DataNormalizer` - Standardizes data formats

**Schema:**
```sql
-- Normalized message storage
CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    sender_id TEXT,
    content TEXT,
    timestamp INTEGER,
    is_from_me BOOLEAN,
    source TEXT,  -- 'imessage', 'email', etc.
    raw_data JSON,
    created_at INTEGER,
    FOREIGN KEY (sender_id) REFERENCES people(id)
);

-- Contact/Person storage
CREATE TABLE people (
    id TEXT PRIMARY KEY,
    name TEXT,
    phone_numbers JSON,
    emails JSON,
    organization TEXT,
    job_title TEXT,
    notes TEXT,
    source TEXT,  -- 'contacts', 'manual'
    last_synced INTEGER
);

-- Sync state tracking
CREATE TABLE sync_state (
    source TEXT PRIMARY KEY,
    last_sync INTEGER,
    last_id TEXT,
    status TEXT
);
```

### Phase 2: People Graph (Relationship Intelligence)

**Goal:** Build and maintain relationship context for all contacts.

**Components:**
- [ ] `PersonEnricher` - Enhances contacts with interaction data
- [ ] `RelationshipScorer` - Calculates relationship strength
- [ ] `InteractionTracker` - Logs all interactions per person
- [ ] `PeopleGraphStore` - Persistent storage with lazy loading

**Features:**
```python
class PeopleGraph:
    def enrich_contact(self, contact_id: str):
        """Enrich contact with message and calendar data."""
        messages = self.get_messages_with(contact_id)
        meetings = self.get_meetings_with(contact_id)

        return {
            'message_count': len(messages),
            'last_message': messages[0].timestamp if messages else None,
            'meeting_count': len(meetings),
            'topics': self.extract_topics(messages),
            'relationship_score': self.calculate_score(contact_id),
            'communication_style': self.analyze_style(messages)
        }

    def get_context_for_person(self, person_id: str) -> str:
        """Generate context string for a person."""
        person = self.get_person(person_id)
        recent = self.get_recent_interactions(person_id, limit=5)

        return f"""
        {person.name} ({person.relationship_type})
        - Relationship strength: {person.relationship_score:.0%}
        - Last contact: {person.last_contact}
        - Common topics: {', '.join(person.topics_discussed[:5])}
        - Recent: {self.summarize_interactions(recent)}
        """
```

### Phase 3: Semantic Layer (Embedding & Search)

**Goal:** Enable semantic search across all temporal data.

**Components:**
- [ ] `MessageEmbedder` - Embeds message chunks using MiniLM-L6
- [ ] `ConversationChunker` - Groups messages into semantic units
- [ ] `SemanticIndex` - FAISS-based vector search
- [ ] `HybridSearch` - Combines semantic + keyword search

**Integration with Existing Services:**
```python
# Reuse existing EmbeddingService

from backend.services.embedding_service import EmbeddingService

class LifeOSEmbedder:
    def __init__(self):
        # Use existing embedding service (MiniLM-L6, 384-dim)
        self.embedding_service = EmbeddingService()

    def embed_messages(self, messages: List[Message]) -> np.ndarray:
        """Embed messages using shared embedding service."""
        chunks = self.chunker.chunk_messages(messages)
        texts = [chunk.to_text() for chunk in chunks]

        # Batch embed for efficiency
        embeddings = self.embedding_service.embed_batch(texts)
        return embeddings

    def search(self, query: str, filters: dict = None) -> List[SearchResult]:
        """Semantic search with optional filters."""
        query_embedding = self.embedding_service.embed(query)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k=50)

        # Apply filters (person, date range, etc.)
        results = self.apply_filters(indices, filters)

        return results[:10]
```

### Phase 4: Query Router (Intelligent Context Assembly)

**Goal:** Route queries to appropriate data sources and assemble context.

**Components:**
- [ ] `IntentClassifier` - Determines query type
- [ ] `EntityExtractor` - Extracts people, dates, topics
- [ ] `ContextAssembler` - Builds context from multiple sources
- [ ] `QueryRouter` - Orchestrates the full query flow

**Query Types:**
```python
class QueryIntent(Enum):
    SEARCH = "search"           # "What did X say about Y?"
    PERSON_LOOKUP = "person"    # "Tell me about John"
    MEETING_PREP = "meeting"    # "Prepare me for my 2pm meeting"
    DRAFT = "draft"             # "Draft a message to Sarah about..."
    TIMELINE = "timeline"       # "What happened last week with project X?"
    SUMMARY = "summary"         # "Summarize my conversations with the team"

class QueryRouter:
    def route(self, query: str) -> QueryResult:
        # 1. Classify intent
        intent = self.intent_classifier.classify(query)

        # 2. Extract entities
        entities = self.entity_extractor.extract(query)

        # 3. Route to appropriate handler
        if intent == QueryIntent.SEARCH:
            return self.handle_search(query, entities)
        elif intent == QueryIntent.PERSON_LOOKUP:
            return self.handle_person_lookup(entities['person'])
        elif intent == QueryIntent.MEETING_PREP:
            return self.handle_meeting_prep(entities['meeting'])
        elif intent == QueryIntent.DRAFT:
            return self.handle_draft(query, entities)
        # ... etc

    def handle_meeting_prep(self, meeting: CalendarEvent) -> str:
        """Generate meeting preparation context."""
        attendees = meeting.attendees

        context_parts = []
        for person_id in attendees:
            # Get person context
            person_ctx = self.people_graph.get_context_for_person(person_id)
            context_parts.append(person_ctx)

            # Get recent relevant messages
            messages = self.search_messages(
                query=meeting.title,
                person_filter=person_id,
                limit=5
            )
            if messages:
                context_parts.append(f"Recent discussions with {person_id}:")
                context_parts.extend([m.content for m in messages])

        return self.context_assembler.assemble(
            meeting_info=meeting,
            attendee_context=context_parts,
            token_budget=2000
        )
```

---

## 7. API Endpoints

### Sync Endpoints

```python
# POST /api/lifeos/sync/messages
# Sync iMessage database

@router.post("/sync/messages")
async def sync_messages(
    since: Optional[datetime] = None,
    full_sync: bool = False
):
    """
    Sync messages from iMessage.

    Args:
        since: Only sync messages after this time (incremental)
        full_sync: Force full re-sync (ignores since)

    Returns:
        SyncResult with count of new/updated messages
    """
    if full_sync:
        since = None
    elif since is None:
        since = sync_state.get_last_sync("messages")

    messages = message_reader.get_messages_since(since)

    # Process and embed in batches
    for batch in chunk(messages, 100):
        embeddings = embedder.embed_messages(batch)
        store.add_messages(batch, embeddings)

    sync_state.update("messages", datetime.now())

    return {"synced": len(messages), "status": "complete"}


# POST /api/lifeos/sync/contacts
# Sync contacts from AddressBook

@router.post("/sync/contacts")
async def sync_contacts():
    """
    Full sync of contacts from AddressBook.
    Contacts don't have incremental sync - always full.
    """
    contacts = contact_reader.get_all_contacts()

    for contact in contacts:
        # Enrich with interaction data
        enriched = people_graph.enrich_contact(contact)
        store.upsert_person(enriched)

    return {"synced": len(contacts), "status": "complete"}


# POST /api/lifeos/sync/calendar
# Sync calendar events

@router.post("/sync/calendar")
async def sync_calendar(
    start: datetime = Query(default_factory=lambda: datetime.now() - timedelta(days=30)),
    end: datetime = Query(default_factory=lambda: datetime.now() + timedelta(days=90))
):
    """
    Sync calendar events in date range.
    Default: 30 days back, 90 days forward.
    """
    events = calendar_reader.get_events(start, end)

    for event in events:
        # Link attendees to people graph
        event.attendees = [
            people_graph.find_person_by_email(a)
            for a in event.raw_attendees
        ]

        # Embed for search
        embedding = embedder.embed_event(event)
        store.upsert_event(event, embedding)

    return {"synced": len(events), "status": "complete"}
```

### Search Endpoints

```python
# GET /api/lifeos/search
# Semantic search across all data

@router.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    sources: List[str] = Query(default=["messages", "calendar", "people"]),
    person: Optional[str] = Query(default=None, description="Filter by person ID"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(default=10, le=50)
):
    """
    Semantic search across Life OS data.

    Args:
        q: Natural language query
        sources: Data sources to search
        person: Optional filter by person
        start_date/end_date: Optional date range filter
        limit: Max results to return

    Returns:
        SearchResults with ranked matches
    """
    results = query_router.search(
        query=q,
        sources=sources,
        filters={
            "person": person,
            "date_range": (start_date, end_date) if start_date else None
        },
        limit=limit
    )

    return {
        "query": q,
        "results": results,
        "total": len(results)
    }
```

### Person Endpoints

```python
# GET /api/lifeos/person/{person_id}
# Get full context for a person

@router.get("/person/{person_id}")
async def get_person(
    person_id: str,
    include_messages: bool = Query(default=True),
    include_meetings: bool = Query(default=True),
    message_limit: int = Query(default=10)
):
    """
    Get comprehensive context for a person.

    Returns:
        PersonContext with profile, relationship data,
        recent messages, and upcoming meetings
    """
    person = people_graph.get_person(person_id)

    if not person:
        raise HTTPException(404, "Person not found")

    result = {
        "person": person.to_dict(),
        "relationship": {
            "score": person.relationship_score,
            "type": person.relationship_type,
            "first_contact": person.first_contact,
            "last_contact": person.last_contact
        }
    }

    if include_messages:
        result["recent_messages"] = store.get_messages_with(
            person_id, limit=message_limit
        )

    if include_meetings:
        result["upcoming_meetings"] = store.get_meetings_with(
            person_id, future_only=True
        )

    return result


# GET /api/lifeos/person/{person_id}/style
# Get communication style analysis

@router.get("/person/{person_id}/style")
async def get_communication_style(person_id: str):
    """
    Analyze communication patterns with this person.
    Used for drafting contextually appropriate messages.
    """
    messages = store.get_messages_with(person_id, limit=100)

    analysis = style_analyzer.analyze(messages)

    return {
        "person_id": person_id,
        "their_style": analysis.their_style,
        "my_style_with_them": analysis.my_style,
        "formality_level": analysis.formality,
        "avg_message_length": analysis.avg_length,
        "common_greetings": analysis.greetings,
        "common_signoffs": analysis.signoffs,
        "response_patterns": analysis.response_patterns
    }
```

### Meeting Prep Endpoints

```python
# POST /api/lifeos/prepare-meeting
# Generate meeting preparation brief

@router.post("/prepare-meeting")
async def prepare_meeting(
    meeting_id: Optional[str] = None,
    meeting_time: Optional[datetime] = None,
    attendees: Optional[List[str]] = None
):
    """
    Generate a comprehensive meeting preparation brief.

    Can specify meeting by:
    - meeting_id: Existing calendar event ID
    - meeting_time: Find meeting at this time
    - attendees: Generate prep for ad-hoc meeting with these people

    Returns:
        MeetingPrep with attendee context, relevant history,
        suggested talking points, and open items
    """
    # Find the meeting
    if meeting_id:
        meeting = store.get_event(meeting_id)
    elif meeting_time:
        meeting = store.find_meeting_at(meeting_time)
    elif attendees:
        meeting = CalendarEvent(
            title="Ad-hoc Meeting",
            attendees=attendees,
            start_time=datetime.now()
        )
    else:
        # Default to next meeting
        meeting = store.get_next_meeting()

    if not meeting:
        raise HTTPException(404, "No meeting found")

    # Generate prep
    prep = query_router.handle_meeting_prep(meeting)

    return {
        "meeting": meeting.to_dict(),
        "attendee_briefs": prep.attendee_briefs,
        "relevant_context": prep.relevant_messages,
        "suggested_topics": prep.suggested_topics,
        "open_items": prep.open_items,
        "last_meeting_notes": prep.last_meeting_summary
    }


# GET /api/lifeos/upcoming
# Get upcoming meetings with prep status

@router.get("/upcoming")
async def get_upcoming(hours: int = Query(default=24)):
    """
    Get upcoming meetings in the next N hours.
    Includes quick context for each.
    """
    meetings = calendar_reader.get_upcoming_meetings(hours)

    result = []
    for meeting in meetings:
        quick_context = query_router.get_quick_context(meeting)
        result.append({
            "meeting": meeting.to_dict(),
            "attendee_count": len(meeting.attendees),
            "has_prep": quick_context.has_relevant_history,
            "quick_context": quick_context.summary
        })

    return {"meetings": result}
```

### Draft Endpoints

```python
# POST /api/lifeos/draft
# Generate contextual message draft

@router.post("/draft")
async def generate_draft(
    recipient: str = Body(..., description="Person ID to message"),
    topic: str = Body(..., description="What the message is about"),
    tone: Optional[str] = Body(default=None, description="Override tone"),
    context: Optional[str] = Body(default=None, description="Additional context")
):
    """
    Generate a message draft using relationship context
    and communication style matching.

    Returns:
        Draft message matching your typical style with this person
    """
    person = people_graph.get_person(recipient)
    style = style_analyzer.get_style_for(recipient)

    # Get relevant conversation history
    history = store.get_messages_with(recipient, limit=20)

    # Generate draft
    draft = draft_generator.generate(
        recipient=person,
        topic=topic,
        style=style,
        tone=tone or style.default_tone,
        history=history,
        additional_context=context
    )

    return {
        "draft": draft.content,
        "style_used": style.to_dict(),
        "confidence": draft.confidence
    }
```

---

## 8. File Structure

```
backend/
├── services/
│   └── lifeos/
│       ├── __init__.py
│       ├── ios/
│       │   ├── __init__.py
│       │   ├── message_reader.py      # iMessage extraction
│       │   ├── contact_reader.py      # Contacts extraction
│       │   └── calendar_reader.py     # Calendar via EventKit
│       ├── sync/
│       │   ├── __init__.py
│       │   ├── sync_manager.py        # Orchestrates syncing
│       │   └── data_normalizer.py     # Standardizes formats
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── people_graph.py        # Relationship graph
│       │   ├── person_enricher.py     # Contact enrichment
│       │   └── relationship_scorer.py # Scoring algorithm
│       ├── semantic/
│       │   ├── __init__.py
│       │   ├── embedder.py            # Message embedding
│       │   ├── chunker.py             # Conversation chunking
│       │   └── search.py              # Semantic search
│       ├── query/
│       │   ├── __init__.py
│       │   ├── router.py              # Query routing
│       │   ├── intent_classifier.py   # Intent detection
│       │   └── context_assembler.py   # Context building
│       └── memory_manager.py          # Memory management
├── routes/
│   └── lifeos.py                      # API endpoints

memory/
└── lifeos/
    ├── identity/
    │   ├── profile.yaml
    │   ├── communication_style.yaml
    │   └── preferences.yaml
    └── data/
        ├── messages.db                # Normalized messages
        ├── people.db                  # People graph
        ├── embeddings.index           # FAISS index
        └── sync_state.json            # Sync tracking

scripts/
└── calendar_helper.swift              # EventKit access helper
```

---

## 9. Security & Privacy

### Data Protection

1. **Local Only**: All data stays on the Mac Mini - no cloud sync
2. **Encrypted Storage**: SQLite databases use SQLCipher encryption
3. **No Logging**: Sensitive content never written to logs
4. **Access Control**: Full Disk Access required, granted per-app

### Sensitive Data Handling

```python
class SensitiveDataFilter:
    """Filter sensitive content before storage/display."""

    PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',           # SSN
        r'\b\d{16}\b',                        # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email (optional)
        r'password[:\s]+\S+',                 # Passwords
    ]

    def filter(self, text: str) -> str:
        """Redact sensitive patterns."""
        for pattern in self.PATTERNS:
            text = re.sub(pattern, '[REDACTED]', text, flags=re.IGNORECASE)
        return text
```

---

## 10. Future Enhancements

### Planned Features

- **Email Integration**: Connect to Mail.app or Gmail for email context
- **Slack/Teams Integration**: Work communication context
- **Voice Memo Transcription**: Index voice memos
- **Photo Context**: Extract text/faces from photos for context
- **Proactive Insights**: "You haven't talked to X in 30 days"
- **Relationship Suggestions**: "Based on your calendar, you might want to reach out to..."

### Performance Optimizations

- **Background Sync**: Continuous incremental sync via launchd
- **Predictive Caching**: Pre-load likely-needed people before meetings
- **Query Caching**: Cache frequent query patterns
- **Embedding Batching**: Queue embeddings for off-peak processing
