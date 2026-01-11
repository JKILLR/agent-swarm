# MYND v3 Architecture Review

**Date:** 2026-01-05
**Reviewer:** Architecture Agent
**Status:** READ-ONLY Review

---

## Executive Summary

MYND v3 is a complex client-server application combining a Three.js-based mind mapping frontend with a Python ML backend ("Brain Server"). The architecture has grown organically, resulting in significant technical debt, particularly around:

1. **Massive inline code** - HTML files contain 10K-45K lines
2. **Configuration duplication** - Brain server URL resolved in 10+ places
3. **Script loading race conditions** - Potential timing issues with initialization
4. **Fragmented URL parameter handling** - `?brain=` param not consistently propagated

---

## Architecture Diagram

```
                           MYND v3 ARCHITECTURE
    ================================================================================

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                              FRONTEND (Browser)                              │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                              │
    │   ┌──────────────────────┐     ┌──────────────────────┐                     │
    │   │   self-dev.html      │     │    index.html        │                     │
    │   │   (376KB, 10K lines) │     │   (1.9MB, 45K lines) │                     │
    │   │                      │     │                      │                     │
    │   │  - CSS (~8000 lines) │     │  - CSS (inline)      │                     │
    │   │  - HTML structure    │     │  - HTML structure    │                     │
    │   │  - Inline scripts    │     │  - Inline scripts    │                     │
    │   └──────────┬───────────┘     └──────────┬───────────┘                     │
    │              │                            │                                  │
    │              └─────────────┬──────────────┘                                  │
    │                            │                                                 │
    │                            ▼                                                 │
    │   ┌────────────────────────────────────────────────────────────────────┐    │
    │   │                     SCRIPT LOADING ORDER                            │    │
    │   │                                                                     │    │
    │   │  1. supabase-js@2 (CDN) ──────────────────────────────────────────▶│    │
    │   │  2. js/config.js (232 lines) ─────────────────────────────────────▶│    │
    │   │     └── Defines CONFIG.BRAIN_SERVER_URL (IIFE, reads ?brain=)      │    │
    │   │  3. js/goal-system.js (1097 lines) ───────────────────────────────▶│    │
    │   │  4. js/local-brain-client.js (2455 lines) ────────────────────────▶│    │
    │   │     └── LocalBrain.serverUrl = CONFIG.BRAIN_SERVER_URL             │    │
    │   │  5. mynd-app-data.js ─────────────────────────────────────────────▶│    │
    │   │  6. [inline script] - DOMContentLoaded → LocalBrain.init()         │    │
    │   │  7. js/reflection-daemon.js (3563 lines) ─────────────────────────▶│    │
    │   │  8. js/reflection-ui.js (970 lines) ──────────────────────────────▶│    │
    │   │  9. js/map-maintenance-daemon.js (1328 lines) ────────────────────▶│    │
    │   │ 10. js/app-module.js (46734 lines) [ES MODULE] ───────────────────▶│    │
    │   │                                                                     │    │
    │   └────────────────────────────────────────────────────────────────────┘    │
    │                                                                              │
    │   ┌──────────────────────────────────────────────────────────────────┐      │
    │   │                    LocalBrain (Client)                            │      │
    │   │                                                                   │      │
    │   │  serverUrl resolution:                                           │      │
    │   │   1. Initial: CONFIG.BRAIN_SERVER_URL (from ?brain= or default)  │      │
    │   │   2. init(): window.MYND_BRAIN_URL > CONFIG > 'localhost:8420'   │      │
    │   │                                                                   │      │
    │   │  ⚠️  ISSUE: serverUrl set TWICE with different logic!             │      │
    │   │                                                                   │      │
    │   │  Methods: init(), chat(), embed(), predictConnections(),         │      │
    │   │           syncMap(), getBrainContext(), transcribe(), etc.       │      │
    │   └─────────────────────────────────┬────────────────────────────────┘      │
    │                                     │                                        │
    └─────────────────────────────────────┼────────────────────────────────────────┘
                                          │
                                          │ HTTP (fetch)
                                          │ POST /health, /brain/chat, /embed, etc.
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        BACKEND (Python/FastAPI)                             │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                              │
    │   ┌──────────────────────────────────────────────────────────────────┐      │
    │   │                    server.py (221KB, ~6000 lines)                 │      │
    │   │                                                                   │      │
    │   │  Config:                                                          │      │
    │   │   PORT = MYND_BRAIN_PORT env or 8420                             │      │
    │   │   HOST = MYND_BRAIN_HOST env or 0.0.0.0                          │      │
    │   │                                                                   │      │
    │   │  Endpoints:                                                       │      │
    │   │   /health (POST)      - Health check                             │      │
    │   │   /embed (POST)       - Text embeddings                          │      │
    │   │   /brain/chat (POST)  - Chat with Claude                         │      │
    │   │   /brain/context (POST) - Get unified context                    │      │
    │   │   /predict/* (POST)   - Graph predictions                        │      │
    │   │   /map/* (POST/GET)   - Map sync/analyze                         │      │
    │   │   /voice/* (POST)     - Whisper transcription                    │      │
    │   │   /image/* (POST)     - CLIP image analysis                      │      │
    │   │   /unified/* (POST/GET) - Unified storage                        │      │
    │   └──────────────────────────────────┬───────────────────────────────┘      │
    │                                      │                                       │
    │   ┌──────────────────────────────────┴───────────────────────────────┐      │
    │   │                         MODULES                                   │      │
    │   │                                                                   │      │
    │   │  brain/                          models/                          │      │
    │   │   ├── unified_brain.py (120KB)   ├── embeddings.py (4KB)         │      │
    │   │   │    - UnifiedBrain            │    - EmbeddingEngine          │      │
    │   │   │    - SelfAwareness           ├── graph_transformer.py (37KB) │      │
    │   │   │    - MemorySystem            │    - MYNDGraphTransformer     │      │
    │   │   │    - PredictionTracker       ├── voice.py (5KB)              │      │
    │   │   │    - KnowledgeDistiller      │    - VoiceTranscriber         │      │
    │   │   │    - MetaLearner             ├── vision.py (10KB)            │      │
    │   │   │    - SelfImprover            │    - VisionEngine             │      │
    │   │   └── context_synthesizer.py     ├── living_asa.py (109KB)       │      │
    │   │        (68KB)                    ├── map_vector_db.py (27KB)     │      │
    │   │        - ContextSynthesizer      ├── conversation_archive.py     │      │
    │   │                                  └── knowledge_extractor.py      │      │
    │   │                                                                   │      │
    │   │  utils/                          data/                            │      │
    │   │   └── cli_executor.py            ├── graph/graph.json            │      │
    │   │       - call_claude_cli_*        ├── conversations/*.json        │      │
    │   │                                  └── learning/*.json             │      │
    │   └──────────────────────────────────────────────────────────────────┘      │
    │                                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Overall Structure Analysis

### Directory Structure

```
mynd-v3/
├── index.html              # Main app (1.9MB, 45K lines)
├── self-dev.html           # Dev mode (376KB, 10K lines)
├── js/
│   ├── config.js           # Configuration (232 lines)
│   ├── local-brain-client.js # Brain client (2455 lines)
│   ├── app-module.js       # Main app logic (46734 lines!)
│   ├── goal-system.js      # Goals (1097 lines)
│   ├── reflection-daemon.js # Reflection (3563 lines)
│   ├── reflection-ui.js    # Reflection UI (970 lines)
│   ├── map-maintenance-daemon.js # Maintenance (1328 lines)
│   └── demo-map.js         # Demo data (2701 lines)
├── mynd-brain/
│   ├── server.py           # FastAPI server (221KB)
│   ├── brain/              # Core brain modules
│   ├── models/             # ML models
│   ├── utils/              # Utilities
│   └── data/               # Persistent data
└── mynd-app-data.js        # Embedded map data (270KB)
```

### Key Observations

1. **Monolithic Files**: `app-module.js` is 46,734 lines - extremely difficult to maintain
2. **Duplicated Structure**: `index.html` and `self-dev.html` share most code but are separate files
3. **No Build System**: Raw JS files served directly, no bundling/minification
4. **Mixed Paradigms**: ES modules (`app-module.js`) mixed with classic scripts

---

## 2. Configuration Management Analysis

### BRAIN_SERVER_URL Configuration Flow

```javascript
// js/config.js (lines 18-30) - FIRST resolution
BRAIN_SERVER_URL: (() => {
    // 1. Check URL parameter first
    const urlParams = new URLSearchParams(window.location.search);
    const brainParam = urlParams.get('brain');
    if (brainParam) return brainParam;

    // 2. Check window.MYND_BRAIN_URL
    if (window.MYND_BRAIN_URL) return window.MYND_BRAIN_URL;

    // 3. Default to localhost
    return 'http://localhost:8420';
})(),
```

```javascript
// js/local-brain-client.js (lines 21-23) - SECOND resolution (at load time)
serverUrl: (typeof CONFIG !== 'undefined' && CONFIG.BRAIN_SERVER_URL)
    ? CONFIG.BRAIN_SERVER_URL
    : 'http://localhost:8420',
```

```javascript
// js/local-brain-client.js (lines 45-51) - THIRD resolution (at init time)
async init() {
    // Priority: 1) window.MYND_BRAIN_URL, 2) CONFIG.BRAIN_SERVER_URL, 3) default
    if (window.MYND_BRAIN_URL) {
        this.serverUrl = window.MYND_BRAIN_URL;
    } else if (typeof CONFIG !== 'undefined' && CONFIG.BRAIN_SERVER_URL) {
        this.serverUrl = CONFIG.BRAIN_SERVER_URL;
    }
    // ...
}
```

### **Critical Issue: URL Resolution Happens Multiple Times**

The `serverUrl` is resolved:
1. **At config.js load** - IIFE executes immediately, reads URL params
2. **At local-brain-client.js load** - Property initialization reads CONFIG
3. **At LocalBrain.init()** - Checks again with different priority order

**Potential Bug**: If `window.MYND_BRAIN_URL` is set AFTER config.js loads but BEFORE `init()`, the value would be different than what CONFIG has.

### Scattered URL References

The brain URL is hard-coded in **10+ places** across the codebase:

| File | Lines | Pattern |
|------|-------|---------|
| app-module.js | 20762, 33493, 33864, 34305, 34338, 34728, 42669, 46549 | `window.MYND_BRAIN_URL \|\| 'http://localhost:8420'` |
| reflection-daemon.js | 3117, 3203, 3271 | `window.MYND_BRAIN_URL \|\| 'http://localhost:8420'` |
| reflection-ui.js | 42, 175, 581 | `window.MYND_BRAIN_URL \|\| 'http://localhost:8420'` |
| config.js | 18-30 | IIFE with URL param check |
| local-brain-client.js | 21-23, 45-51 | Multiple resolution points |

**This is a major maintenance problem** - a URL change requires updating 10+ locations.

---

## 3. Data Flow Patterns

### Frontend → LocalBrain → Brain Server Flow

```
User Action
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (app-module.js or reflection-*.js)                    │
│                                                                  │
│  // Direct call (INCONSISTENT - bypasses LocalBrain!)           │
│  const brainUrl = window.MYND_BRAIN_URL || 'http://localhost:8420';
│  fetch(`${brainUrl}/brain/chat`, {...})                         │
│                                                                  │
│  // OR via LocalBrain (CORRECT approach)                        │
│  await LocalBrain.chat(message, history, session)               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  LocalBrain (local-brain-client.js)                             │
│                                                                  │
│  serverUrl = CONFIG.BRAIN_SERVER_URL (from ?brain= or default)  │
│                                                                  │
│  async chat() {                                                  │
│      fetch(`${this.serverUrl}/brain/chat`, {                    │
│          body: JSON.stringify({                                  │
│              user_message,                                       │
│              conversation_history,                               │
│              system_prompt                                       │
│          })                                                      │
│      })                                                          │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
    │
    │ HTTP POST
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Brain Server (server.py)                                        │
│                                                                  │
│  @app.post("/brain/chat")                                        │
│  async def brain_chat(request: BrainChatRequest):               │
│      # 1. Get unified brain context                              │
│      context = await unified_brain.get_context(...)              │
│                                                                  │
│      # 2. Call Claude via CLI                                    │
│      response = await call_claude_cli_with_conversation(...)    │
│                                                                  │
│      # 3. Return response                                        │
│      return {"message": response, "model": "claude"}            │
└─────────────────────────────────────────────────────────────────┘
```

### ?brain= URL Parameter Processing

```
URL: self-dev.html?brain=https://example.com:8420
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  config.js loads (lines 18-30)                                   │
│                                                                  │
│  const urlParams = new URLSearchParams(window.location.search); │
│  const brainParam = urlParams.get('brain');                     │
│  // brainParam = 'https://example.com:8420'                     │
│                                                                  │
│  CONFIG.BRAIN_SERVER_URL = brainParam                           │
│  // CONFIG.BRAIN_SERVER_URL = 'https://example.com:8420'        │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  local-brain-client.js loads (line 21-23)                        │
│                                                                  │
│  LocalBrain.serverUrl = CONFIG.BRAIN_SERVER_URL                 │
│  // LocalBrain.serverUrl = 'https://example.com:8420' ✓         │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  BUT: app-module.js, reflection-*.js use DIFFERENT logic!       │
│                                                                  │
│  const brainUrl = window.MYND_BRAIN_URL || 'localhost:8420'    │
│  // ❌ Does NOT read URL params!                                 │
│  // ❌ Uses hardcoded default!                                   │
│  // window.MYND_BRAIN_URL is typically undefined                │
└─────────────────────────────────────────────────────────────────┘

⚠️  RESULT: LocalBrain uses custom URL, but direct fetch calls use default!
```

### Where URL Values Could Be Cached/Overwritten

| Location | When | Value Source | Issue |
|----------|------|--------------|-------|
| `CONFIG.BRAIN_SERVER_URL` | Script load | `?brain=` param, then `window.MYND_BRAIN_URL`, then default | Computed once via IIFE |
| `LocalBrain.serverUrl` | Script load | `CONFIG.BRAIN_SERVER_URL` | Re-computed at init() |
| Direct fetch calls | Runtime | `window.MYND_BRAIN_URL \|\| default` | Ignores CONFIG and URL params! |

**Root Cause of Connection Issues:**
If the `?brain=` parameter is passed, `CONFIG.BRAIN_SERVER_URL` gets the correct value, but direct fetch calls in `app-module.js` and `reflection-*.js` **ignore CONFIG** and fall back to the default.

---

## 4. Script Loading Order Analysis

### self-dev.html Script Loading (lines 10223-10271)

```html
<!-- Order of execution -->

1. <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
   <!-- External dependency, must load first -->

2. <script src="js/config.js"></script>
   <!-- ✓ Defines CONFIG global -->
   <!-- ✓ BRAIN_SERVER_URL computed here (reads URL params) -->

3. <script src="js/goal-system.js"></script>
   <!-- Goal management, no dependencies on LocalBrain -->

4. <script src="js/local-brain-client.js"></script>
   <!-- ✓ Can read CONFIG.BRAIN_SERVER_URL -->
   <!-- ✓ Registers DOMContentLoaded handler for init() -->

5. <script src="mynd-app-data.js"></script>
   <!-- Embedded map data -->

6. <script> /* inline DOMContentLoaded handler */ </script>
   <!-- Calls LocalBrain.init() when DOM ready -->

7. <script type="importmap">...</script>
   <!-- Three.js import map -->

8. <script src="js/reflection-daemon.js"></script>
   <!-- ⚠️ Uses hardcoded URL fallback, not LocalBrain -->

9. <script src="js/reflection-ui.js"></script>
   <!-- ⚠️ Uses hardcoded URL fallback, not LocalBrain -->

10. <script src="js/map-maintenance-daemon.js"></script>
    <!-- Map maintenance -->

11. <script type="module" src="js/app-module.js?v=11"></script>
    <!-- ES Module - executes after all classic scripts -->
    <!-- ⚠️ Contains many hardcoded URL fallbacks -->
```

### Race Conditions Identified

#### Race Condition 1: LocalBrain.init() vs Direct Fetch Calls

```
Timeline:
─────────────────────────────────────────────────────────────────────────────
DOMContentLoaded fires
    │
    ├──▶ LocalBrain.init() starts (async)
    │        │
    │        └──▶ checkAvailability() - takes 0-5 seconds
    │
    ├──▶ app-module.js may call fetch() directly
    │        │
    │        └──▶ Uses hardcoded URL (not LocalBrain.serverUrl!)
    │
    └──▶ reflection-daemon.js may call fetch() directly
             │
             └──▶ Uses hardcoded URL (not LocalBrain.serverUrl!)

⚠️ If app-module.js/reflection-*.js make requests before LocalBrain.init()
   completes, they use the wrong URL!
```

#### Race Condition 2: ES Module vs Classic Script Timing

```
local-brain-client.js (classic script):
    └──▶ Sets up LocalBrain object
    └──▶ Registers DOMContentLoaded for init()

app-module.js (ES module):
    └──▶ Deferred execution (after HTML parsing)
    └──▶ May access LocalBrain before init() completes

⚠️ ES modules are deferred by default - LocalBrain may not be
   fully initialized when app-module.js first accesses it.
```

#### Race Condition 3: Auto-Init at Script Load

```javascript
// local-brain-client.js line 2448
document.addEventListener('DOMContentLoaded', () => LocalBrain.init());
```

This auto-init competes with the inline script's manual init:

```html
<!-- self-dev.html lines 10249-10258 -->
<script>
    document.addEventListener('DOMContentLoaded', function() {
        if (typeof LocalBrain !== 'undefined') {
            LocalBrain.init().then(connected => {...});
        }
    });
</script>
```

**Both handlers run**, potentially calling `init()` twice.

---

## 5. Code Organization Issues

### File Size Problems

| File | Size | Lines | Issue |
|------|------|-------|-------|
| index.html | 1.9 MB | 44,828 | **Massive** - contains all CSS inline |
| app-module.js | 2.1 MB | 46,734 | **Enormous** - should be split into 20+ modules |
| self-dev.html | 376 KB | 10,273 | Large but more manageable |
| unified_brain.py | 121 KB | ~3,500 | Should be split |
| living_asa.py | 109 KB | ~3,000 | Should be split |

### Excessive Inline Code

**index.html** contains:
- ~8,000 lines of CSS
- ~35,000 lines of embedded JavaScript
- All HTML structure

**self-dev.html** contains:
- ~8,000 lines of CSS (duplicated from index.html)
- ~1,500 lines of inline JavaScript
- HTML structure

### What Should Be Modularized

```
Current Structure (Problematic):
─────────────────────────────────
app-module.js (46,734 lines)
    ├── Three.js scene management
    ├── Node/Map rendering
    ├── UI components
    ├── Event handlers
    ├── AI/Chat integration
    ├── Storage/IndexedDB
    ├── Import/Export
    ├── Attachment handling
    ├── Neural network (TensorFlow.js)
    └── Everything else...

Recommended Structure:
─────────────────────────────────
js/
├── core/
│   ├── config.js          (centralized config)
│   ├── events.js          (event bus)
│   └── storage.js         (IndexedDB/localStorage)
├── brain/
│   ├── local-brain.js     (LocalBrain client)
│   ├── neural-net.js      (TensorFlow.js)
│   └── ai-chat.js         (Chat/completion handling)
├── scene/
│   ├── scene-manager.js   (Three.js setup)
│   ├── node-renderer.js   (Node visualization)
│   ├── camera-controls.js (Camera/orbit)
│   └── labels.js          (Label rendering)
├── ui/
│   ├── panels.js          (Side panels)
│   ├── modals.js          (Modal dialogs)
│   ├── chat-ui.js         (Chat interface)
│   └── toolbar.js         (Toolbar buttons)
├── map/
│   ├── map-store.js       (Map data management)
│   ├── node-operations.js (CRUD operations)
│   └── import-export.js   (Import/export)
└── main.js                (Entry point)
```

### Duplicated Code Patterns

The following pattern appears in **9 different locations**:

```javascript
const brainUrl = window.MYND_BRAIN_URL || 'http://localhost:8420';
fetch(`${brainUrl}/some/endpoint`, {...})
```

Should be replaced with:

```javascript
// Use LocalBrain consistently
await LocalBrain.someMethod(...)

// OR centralize URL resolution
import { getBrainUrl } from './core/config.js';
const brainUrl = getBrainUrl();
```

---

## 6. Recommendations for LocalBrain Connection Issue

### Immediate Fixes

1. **Centralize URL Resolution**

```javascript
// js/config.js - Add method to get current URL
CONFIG.getBrainUrl = function() {
    // Single source of truth
    if (window.MYND_BRAIN_URL) return window.MYND_BRAIN_URL;
    return this.BRAIN_SERVER_URL;
};
```

2. **Fix LocalBrain Double Init**

```javascript
// local-brain-client.js - Remove auto-init, let callers control it
// DELETE: document.addEventListener('DOMContentLoaded', () => LocalBrain.init());

// Add initialization guard
let _initPromise = null;
async init() {
    if (_initPromise) return _initPromise;
    _initPromise = this._doInit();
    return _initPromise;
}
```

3. **Replace All Direct Fetch Calls**

In `app-module.js`, `reflection-daemon.js`, `reflection-ui.js`:

```javascript
// BEFORE (scattered across 10+ places)
const brainUrl = window.MYND_BRAIN_URL || 'http://localhost:8420';
const res = await fetch(`${brainUrl}/brain/chat`, {...});

// AFTER (use LocalBrain)
const res = await LocalBrain.chat(message, history, session);
```

4. **Propagate URL Param to window.MYND_BRAIN_URL**

```javascript
// config.js - Set window global for other scripts
if (CONFIG.BRAIN_SERVER_URL !== 'http://localhost:8420') {
    window.MYND_BRAIN_URL = CONFIG.BRAIN_SERVER_URL;
}
```

### Architecture Improvements

1. **Build System**: Introduce Vite/Webpack to:
   - Bundle and minify JS
   - Extract CSS into separate files
   - Tree-shake unused code
   - Enable code splitting

2. **Module System**: Convert to ES modules:
   - Split `app-module.js` into 15-20 focused modules
   - Use proper imports/exports
   - Enable better tooling support

3. **Single LocalBrain Instance**: Ensure all code paths use `LocalBrain`:
   - Remove all direct `fetch()` calls to brain server
   - LocalBrain handles URL resolution, retries, fallbacks

4. **Configuration Injection**: Pass brain URL at startup:
   ```javascript
   // main.js
   const brainUrl = new URLSearchParams(location.search).get('brain')
       || window.MYND_BRAIN_URL
       || 'http://localhost:8420';

   LocalBrain.configure({ serverUrl: brainUrl });
   await LocalBrain.init();
   ```

---

## 7. Summary of Critical Issues

| Priority | Issue | Impact | Location |
|----------|-------|--------|----------|
| **HIGH** | URL resolution happens in 10+ places | Wrong server used | Throughout codebase |
| **HIGH** | Direct fetch bypasses LocalBrain | Config ignored | app-module.js, reflection-*.js |
| **HIGH** | Double init() calls | Potential race | local-brain-client.js + inline |
| **MEDIUM** | 46K line app-module.js | Unmaintainable | js/app-module.js |
| **MEDIUM** | Duplicated CSS in HTML files | Wasted bandwidth | index.html, self-dev.html |
| **LOW** | No build system | No optimization | Entire frontend |

---

## Appendix: Key File Locations

### Configuration
- `js/config.js:18-30` - BRAIN_SERVER_URL IIFE
- `js/local-brain-client.js:21-23` - serverUrl property
- `js/local-brain-client.js:44-64` - init() method

### Direct Fetch Calls (Need Fixing)
- `js/app-module.js:20762`
- `js/app-module.js:33493`
- `js/app-module.js:33864`
- `js/app-module.js:34305`
- `js/app-module.js:34338`
- `js/app-module.js:34728`
- `js/app-module.js:42669`
- `js/app-module.js:46549`
- `js/reflection-daemon.js:3117`
- `js/reflection-daemon.js:3203`
- `js/reflection-daemon.js:3271`
- `js/reflection-ui.js:42`
- `js/reflection-ui.js:175`
- `js/reflection-ui.js:581`

### Script Loading
- `self-dev.html:10223-10271` - Script tag order

---

*Review completed: 2026-01-05*
