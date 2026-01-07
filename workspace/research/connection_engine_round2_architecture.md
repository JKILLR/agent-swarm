# Connection Engine v0.1 - Refined Architecture

**Date**: 2026-01-07
**Focus**: Ruthlessly minimal MVP that demonstrates value on 8GB Mac Mini
**Philosophy**: Ship something that works today, expand tomorrow

---

## Constraint Reality Check

**8GB Mac Mini means:**
- Embedding model already loaded: ~400MB (all-MiniLM-L6-v2)
- Leave 2-3GB for LLM API calls, OS, frontend
- No additional ML models in memory
- Background jobs must be non-blocking and lightweight
- Prefer batch operations over real-time processing

**What we already have:**
- `EmbeddingService` - singleton, lazy-loaded, 384-dim vectors
- `SemanticIndex` - vector storage with cosine similarity search
- `MindGraph` - hierarchical node storage with edges
- `JobManager` - SQLite-backed job queue with progress tracking
- `MemoryStore` - simple key-value facts storage

---

## v0.1 Scope: "Connections That Matter"

### Features INCLUDED
1. **Session Co-occurrence** - Track what topics appear together in a session
2. **Cross-Session Resonance** - Find patterns that repeat across days
3. **Simple Bridge Detection** - Identify nodes that connect different topic clusters
4. **Proactive Surfacing** - Surface high-resonance patterns when starting a session

### Features CUT (v0.2+)
- ~~Multi-lens embeddings~~ (4x embedding cost)
- ~~Full dialogue crystallization~~ (4 LLM calls per ingestion)
- ~~Real-time challenger agent~~ (async re-challenge is enough)
- ~~Tension detection~~ (complex, can wait)
- ~~Sophisticated session boundary detection~~ (use simple time gaps)

---

## Data Model

### New Tables (SQLite, same DB as jobs.db)

```sql
-- Track what appears together in a session
CREATE TABLE co_occurrences (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    node_a_id TEXT NOT NULL,
    node_b_id TEXT NOT NULL,
    strength REAL DEFAULT 1.0,       -- proximity-weighted
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    occurrence_count INTEGER DEFAULT 1,
    UNIQUE(session_id, node_a_id, node_b_id)
);

-- Patterns that repeat across sessions
CREATE TABLE resonance_patterns (
    id TEXT PRIMARY KEY,
    signature TEXT NOT NULL,          -- embedding or node_id based
    representative_node_id TEXT,      -- most central node
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    session_count INTEGER DEFAULT 1,  -- how many sessions
    total_occurrences INTEGER DEFAULT 1,
    avg_temporal_gap_hours REAL,      -- time between occurrences
    resonance_score REAL DEFAULT 0.0,
    last_surfaced TEXT,               -- when we showed this to user
    surfaced_count INTEGER DEFAULT 0
);

-- Link patterns to their member nodes/sessions
CREATE TABLE pattern_members (
    pattern_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    PRIMARY KEY (pattern_id, node_id, session_id)
);

-- Simple cluster assignments (for bridge detection)
CREATE TABLE node_clusters (
    node_id TEXT PRIMARY KEY,
    cluster_id INTEGER NOT NULL,
    distance_to_centroid REAL,
    is_bridge BOOLEAN DEFAULT FALSE,
    updated_at TEXT NOT NULL
);

CREATE INDEX idx_co_occ_session ON co_occurrences(session_id);
CREATE INDEX idx_patterns_score ON resonance_patterns(resonance_score DESC);
CREATE INDEX idx_clusters_cluster ON node_clusters(cluster_id);
```

### Memory Layout Estimate
- Co-occurrences: ~100 bytes/row, 1000 rows = 100KB
- Patterns: ~200 bytes/row, 100 patterns = 20KB
- Pattern members: ~80 bytes/row, 500 rows = 40KB
- Node clusters: ~50 bytes/row, 500 nodes = 25KB
- **Total: < 200KB** (trivial)

---

## File Structure

```
backend/
  services/
    connection_engine/
      __init__.py           # exports ConnectionEngine singleton
      models.py             # dataclasses: CoOccurrence, ResonancePattern, etc.
      storage.py            # SQLite operations (ConnectionStore)
      session_tracker.py    # SessionCooccurrenceTracker
      resonance_detector.py # CrossSessionResonanceDetector
      cluster_service.py    # SimpleClusterService (k-means)
      surface_service.py    # ProactiveSurfaceService

  routes/
    connections.py          # API endpoints for COO interaction
```

---

## Class Design

### 1. ConnectionStore (storage.py)
```python
class ConnectionStore:
    """SQLite storage for connection engine data."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    # Co-occurrences
    def upsert_co_occurrence(self, session_id: str, node_a: str, node_b: str, strength: float)
    def get_session_co_occurrences(self, session_id: str) -> list[CoOccurrence]
    def get_node_co_occurrences(self, node_id: str, limit: int = 50) -> list[CoOccurrence]

    # Patterns
    def upsert_pattern(self, pattern: ResonancePattern)
    def get_top_patterns(self, limit: int = 10, min_score: float = 0.3) -> list[ResonancePattern]
    def get_unsurfaced_patterns(self, days_since: int = 7) -> list[ResonancePattern]
    def mark_pattern_surfaced(self, pattern_id: str)

    # Clusters
    def save_clusters(self, assignments: dict[str, int], centroids: dict[int, np.ndarray])
    def get_node_cluster(self, node_id: str) -> Optional[int]
    def get_bridge_nodes(self) -> list[str]
```

### 2. SessionCooccurrenceTracker (session_tracker.py)
```python
class SessionCooccurrenceTracker:
    """Track what appears together within a single session."""

    WINDOW_SIZE = 5  # Consider last 5 nodes as co-occurring
    SESSION_GAP_MINUTES = 30  # New session after 30 min inactivity

    def __init__(self, store: ConnectionStore, graph: MindGraph):
        self.store = store
        self.graph = graph
        self._current_session: Optional[str] = None
        self._session_start: Optional[datetime] = None
        self._recent_nodes: deque[str] = deque(maxlen=self.WINDOW_SIZE)

    def on_node_accessed(self, node_id: str):
        """Called when a node is created, searched for, or referenced."""
        self._ensure_session()

        # Update co-occurrences with recent nodes
        for other_id in self._recent_nodes:
            if other_id != node_id:
                strength = 1.0 / (len(self._recent_nodes) - self._recent_nodes.index(other_id))
                self.store.upsert_co_occurrence(
                    self._current_session, node_id, other_id, strength
                )

        self._recent_nodes.append(node_id)

    def _ensure_session(self):
        """Start new session if needed."""
        now = datetime.now()
        if self._session_start is None or \
           (now - self._session_start).total_minutes() > self.SESSION_GAP_MINUTES:
            self._current_session = f"session-{now.strftime('%Y%m%d_%H%M%S')}"
            self._session_start = now
            self._recent_nodes.clear()
```

### 3. CrossSessionResonanceDetector (resonance_detector.py)
```python
class CrossSessionResonanceDetector:
    """Background job: Find patterns that repeat across sessions."""

    SIMILARITY_THRESHOLD = 0.75
    MIN_SESSIONS = 2  # Need to appear in 2+ sessions

    def __init__(self, store: ConnectionStore, graph: MindGraph, embedding: EmbeddingService):
        self.store = store
        self.graph = graph
        self.embedding = embedding

    async def detect_patterns(self) -> list[ResonancePattern]:
        """Run pattern detection across all sessions."""
        # Get all unique nodes that appeared in co-occurrences
        nodes_by_session = self._get_nodes_by_session()

        # For each pair of sessions, find similar nodes
        patterns = {}
        sessions = list(nodes_by_session.keys())

        for i, session_a in enumerate(sessions[:-1]):
            for session_b in sessions[i+1:]:
                # Find nodes in session_b similar to nodes in session_a
                matches = await self._find_cross_session_matches(
                    nodes_by_session[session_a],
                    nodes_by_session[session_b]
                )

                for node_a, node_b, similarity in matches:
                    # Create or update pattern
                    pattern_key = self._get_pattern_key(node_a, node_b)
                    if pattern_key not in patterns:
                        patterns[pattern_key] = ResonancePattern(
                            id=str(uuid.uuid4()),
                            signature=pattern_key,
                            representative_node_id=node_a,
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                        )

                    pattern = patterns[pattern_key]
                    pattern.session_count += 1
                    pattern.total_occurrences += 1
                    # Calculate temporal gap from session timestamps
                    pattern.update_resonance_score()

        # Save patterns
        for pattern in patterns.values():
            if pattern.session_count >= self.MIN_SESSIONS:
                self.store.upsert_pattern(pattern)

        return list(patterns.values())

    def _calculate_resonance_score(self, pattern: ResonancePattern) -> float:
        """
        Resonance score formula:
        - More sessions = higher score
        - Longer average gap = higher score (ideas that persist)
        - More recent = slight boost
        """
        session_factor = min(pattern.session_count / 5, 1.0)  # Cap at 5 sessions
        gap_factor = min((pattern.avg_temporal_gap_hours or 0) / 168, 1.0)  # Cap at 1 week
        recency_factor = 0.1 if pattern.last_seen > datetime.now() - timedelta(days=7) else 0

        return 0.5 * session_factor + 0.4 * gap_factor + 0.1 + recency_factor
```

### 4. SimpleClusterService (cluster_service.py)
```python
class SimpleClusterService:
    """Simple k-means clustering for bridge detection."""

    def __init__(self, store: ConnectionStore, graph: MindGraph, embedding: EmbeddingService):
        self.store = store
        self.graph = graph
        self.embedding = embedding

    async def update_clusters(self, n_clusters: int = 5):
        """Re-cluster all nodes. Run as background job."""
        # Get all node embeddings from semantic index
        nodes = list(self.graph._nodes.values())
        if len(nodes) < n_clusters:
            return

        # Build embedding matrix
        embeddings = []
        node_ids = []
        for node in nodes:
            emb = self.graph.semantic_index._embeddings.get(node.id)
            if emb is not None:
                embeddings.append(emb)
                node_ids.append(node.id)

        if len(embeddings) < n_clusters:
            return

        X = np.array(embeddings)

        # Simple k-means (sklearn if available, else naive impl)
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_
        except ImportError:
            labels, centroids = self._naive_kmeans(X, n_clusters)

        # Save assignments
        assignments = {node_ids[i]: int(labels[i]) for i in range(len(node_ids))}
        self.store.save_clusters(assignments, {i: centroids[i] for i in range(n_clusters)})

        # Mark bridges (nodes close to 2+ centroids)
        self._mark_bridges(X, node_ids, centroids, labels)

    def _mark_bridges(self, X, node_ids, centroids, labels):
        """Find nodes that are close to multiple cluster centroids."""
        for i, (node_id, emb) in enumerate(zip(node_ids, X)):
            distances = [np.linalg.norm(emb - c) for c in centroids]
            sorted_dists = sorted(enumerate(distances), key=lambda x: x[1])

            # If 2nd closest is within 1.5x of closest, it's a bridge
            if len(sorted_dists) >= 2:
                closest_dist = sorted_dists[0][1]
                second_dist = sorted_dists[1][1]
                if second_dist < closest_dist * 1.5 and closest_dist > 0:
                    self.store.mark_as_bridge(node_id, True)
```

### 5. ProactiveSurfaceService (surface_service.py)
```python
class ProactiveSurfaceService:
    """Generate proactive insights for the user."""

    def __init__(self, store: ConnectionStore, graph: MindGraph):
        self.store = store
        self.graph = graph

    def get_session_start_insights(self) -> list[dict]:
        """Called when user starts a new chat session."""
        insights = []

        # High resonance patterns not recently surfaced
        patterns = self.store.get_unsurfaced_patterns(days_since=7)
        for pattern in patterns[:3]:  # Max 3 patterns
            node = self.graph.get_node(pattern.representative_node_id)
            if node:
                insights.append({
                    "type": "resonance",
                    "title": f"Recurring theme: {node.label}",
                    "description": f"This topic has come up in {pattern.session_count} sessions over the past week.",
                    "pattern_id": pattern.id,
                    "node_id": node.id,
                })
                self.store.mark_pattern_surfaced(pattern.id)

        # Bridge nodes (connect different areas)
        bridge_nodes = self.store.get_bridge_nodes()[:2]  # Max 2 bridges
        for node_id in bridge_nodes:
            node = self.graph.get_node(node_id)
            if node:
                insights.append({
                    "type": "bridge",
                    "title": f"Connecting idea: {node.label}",
                    "description": "This concept bridges multiple areas of your thinking.",
                    "node_id": node.id,
                })

        return insights

    def get_contextual_connections(self, current_node_id: str) -> list[dict]:
        """Get connections relevant to current topic."""
        connections = []

        # Co-occurring nodes from past sessions
        co_occurrences = self.store.get_node_co_occurrences(current_node_id, limit=5)
        for co in co_occurrences:
            other_id = co.node_b_id if co.node_a_id == current_node_id else co.node_a_id
            node = self.graph.get_node(other_id)
            if node:
                connections.append({
                    "type": "co_occurrence",
                    "node": node.to_dict(),
                    "strength": co.strength,
                    "sessions": co.occurrence_count,
                })

        return connections
```

---

## Integration Points

### 1. Hook into MindGraph (mind_graph.py)

```python
# In MindGraph.add_node(), after creating node:
from backend.services.connection_engine import get_connection_engine
engine = get_connection_engine()
engine.on_node_accessed(node.id)

# In MindGraph.semantic_search(), after search:
for result in results:
    engine.on_node_accessed(result.node.id)
```

### 2. Hook into Chat Handler (chat_handler.py)

```python
# At session start, inject insights into system context:
async def get_proactive_context(session_id: str) -> str:
    engine = get_connection_engine()
    insights = engine.get_session_start_insights()

    if not insights:
        return ""

    lines = ["### Proactive Insights"]
    for insight in insights:
        lines.append(f"- **{insight['title']}**: {insight['description']}")

    return "\n".join(lines)
```

### 3. Register Background Jobs (jobs.py)

```python
# Add new job types
class JobType(str, Enum):
    CHAT = "chat"
    SWARM_DIRECTIVE = "swarm_directive"
    RESONANCE_DETECTION = "resonance_detection"  # NEW
    CLUSTER_UPDATE = "cluster_update"            # NEW

# In JobManager._execute_job():
elif job.type == "resonance_detection":
    result = await self._execute_resonance_job(job)
elif job.type == "cluster_update":
    result = await self._execute_cluster_job(job)
```

### 4. Schedule Periodic Jobs

```python
# In main.py startup:
from backend.services.connection_engine import get_connection_engine

@app.on_event("startup")
async def schedule_connection_jobs():
    job_manager = get_job_manager()

    # Run resonance detection every 6 hours
    async def resonance_loop():
        while True:
            await asyncio.sleep(6 * 60 * 60)  # 6 hours
            await job_manager.submit_job("resonance_detection", "Detect cross-session patterns")

    # Run clustering daily
    async def cluster_loop():
        while True:
            await asyncio.sleep(24 * 60 * 60)  # 24 hours
            await job_manager.submit_job("cluster_update", "Update node clusters")

    asyncio.create_task(resonance_loop())
    asyncio.create_task(cluster_loop())
```

---

## API Surface for COO

### routes/connections.py

```python
router = APIRouter(prefix="/api/connections", tags=["connections"])

@router.get("/insights")
async def get_insights() -> dict:
    """Get proactive insights for current session."""
    engine = get_connection_engine()
    return {"insights": engine.surface_service.get_session_start_insights()}

@router.get("/patterns")
async def get_patterns(limit: int = 10, min_score: float = 0.3) -> dict:
    """Get top resonance patterns."""
    engine = get_connection_engine()
    patterns = engine.store.get_top_patterns(limit, min_score)
    return {"patterns": [p.to_dict() for p in patterns]}

@router.get("/nodes/{node_id}/connections")
async def get_node_connections(node_id: str) -> dict:
    """Get connections for a specific node."""
    engine = get_connection_engine()
    return {"connections": engine.surface_service.get_contextual_connections(node_id)}

@router.get("/bridges")
async def get_bridges() -> dict:
    """Get nodes that bridge different topic clusters."""
    engine = get_connection_engine()
    bridge_ids = engine.store.get_bridge_nodes()
    graph = get_mind_graph()
    bridges = [graph.get_node(nid).to_dict() for nid in bridge_ids if graph.get_node(nid)]
    return {"bridges": bridges}

@router.post("/jobs/resonance")
async def trigger_resonance_detection() -> dict:
    """Manually trigger resonance detection."""
    job_manager = get_job_manager()
    job = await job_manager.submit_job("resonance_detection", "Manual resonance detection")
    return {"job_id": job.id, "status": job.status.value}

@router.post("/jobs/cluster")
async def trigger_cluster_update() -> dict:
    """Manually trigger cluster update."""
    job_manager = get_job_manager()
    job = await job_manager.submit_job("cluster_update", "Manual cluster update")
    return {"job_id": job.id, "status": job.status.value}

@router.get("/stats")
async def get_connection_stats() -> dict:
    """Get connection engine statistics."""
    engine = get_connection_engine()
    return {
        "co_occurrence_count": engine.store.count_co_occurrences(),
        "pattern_count": engine.store.count_patterns(),
        "bridge_count": len(engine.store.get_bridge_nodes()),
        "last_resonance_run": engine.store.get_last_job_time("resonance_detection"),
        "last_cluster_run": engine.store.get_last_job_time("cluster_update"),
    }
```

---

## Implementation Order

### Phase 1: Foundation (Day 1)
1. Create `backend/services/connection_engine/` directory structure
2. Implement `models.py` with dataclasses
3. Implement `storage.py` with SQLite operations
4. Implement `__init__.py` with singleton `ConnectionEngine` class

### Phase 2: Real-time Tracking (Day 2)
1. Implement `session_tracker.py`
2. Add hooks to `MindGraph.add_node()` and `MindGraph.semantic_search()`
3. Test co-occurrence tracking

### Phase 3: Background Detection (Day 3)
1. Implement `resonance_detector.py`
2. Implement `cluster_service.py`
3. Register job types in `jobs.py`
4. Add scheduled jobs to startup

### Phase 4: Surfacing (Day 4)
1. Implement `surface_service.py`
2. Implement `routes/connections.py`
3. Hook insights into chat handler
4. Test end-to-end flow

### Phase 5: Polish (Day 5)
1. Add logging and error handling
2. Write basic tests
3. Document API endpoints
4. Performance profiling

---

## Success Metrics for v0.1

1. **Works on 8GB Mac Mini** - No memory growth over 24hrs
2. **Session tracking works** - Co-occurrences recorded for each session
3. **Patterns detected** - At least 1 pattern found after 3+ sessions with similar topics
4. **Bridges identified** - After 50+ nodes, some marked as bridges
5. **Insights surface** - On session start, see relevant patterns
6. **API responds** - All endpoints return data < 100ms

---

## What This Enables

Once v0.1 is working, the COO can:
- Ask "What patterns have you noticed in my thinking?"
- Get contextual suggestions when discussing a topic
- See which ideas connect different areas
- Have the system proactively remind them of recurring themes

This is the foundation for a system that **LEARNS**, not just stores.

---

*Ready for implementation review and Phase 1 kickoff*
