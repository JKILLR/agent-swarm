# MYND LocalBrain Connection Deep Dive Analysis

## Executive Summary

After thorough investigation, **no hardcoded "maple-leisure" or any other trycloudflare.com URL was found in the codebase**. The issue is likely caused by **browser caching** (service worker or HTTP cache) rather than code issues.

---

## 1. URL Parameter Flow Analysis

### config.js IIFE Evaluation (Lines 18-30)

```javascript
BRAIN_SERVER_URL: (() => {
    // Check URL parameter first (for testing/debugging)
    const urlParams = new URLSearchParams(window.location.search);
    const brainParam = urlParams.get('brain');
    if (brainParam) return brainParam;

    // Check for Runpod/cloud deployment environment
    if (window.MYND_BRAIN_URL) return window.MYND_BRAIN_URL;

    // Default to local server
    return 'http://localhost:8420';
})(),
```

**Timing Analysis:**
- The IIFE executes **immediately** when `config.js` is parsed
- This happens during synchronous script loading in `<head>` or early `<body>`
- The URL parameter is read **at script load time**, not at DOMContentLoaded
- `CONFIG.BRAIN_SERVER_URL` is set ONCE and then the object is partially frozen

**Finding: The URL parameter IS correctly captured IF present at load time.**

---

## 2. LocalBrain Initialization Flow

### local-brain-client.js serverUrl Setting

There are **THREE places** where serverUrl can be set:

#### A. Object Declaration (Line 21-23)
```javascript
serverUrl: (typeof CONFIG !== 'undefined' && CONFIG.BRAIN_SERVER_URL)
    ? CONFIG.BRAIN_SERVER_URL
    : 'http://localhost:8420',
```
- Executes at script parse time
- CONFIG is available because config.js loads first
- **This SHOULD get the correct URL from CONFIG**

#### B. init() Method (Lines 44-50)
```javascript
async init() {
    // Priority: 1) window.MYND_BRAIN_URL, 2) CONFIG.BRAIN_SERVER_URL, 3) default
    if (window.MYND_BRAIN_URL) {
        this.serverUrl = window.MYND_BRAIN_URL;
    } else if (typeof CONFIG !== 'undefined' && CONFIG.BRAIN_SERVER_URL) {
        this.serverUrl = CONFIG.BRAIN_SERVER_URL;
    }
    console.log(`🧠 LocalBrain: Initializing with server: ${this.serverUrl}`);
    // ...
}
```

#### C. Auto-initialization (Line 2448)
```javascript
document.addEventListener('DOMContentLoaded', () => LocalBrain.init());
```

**AND** manual initialization in self-dev.html (Line 10251):
```javascript
LocalBrain.init().then(connected => { ... });
```

**Result: LocalBrain.init() is called TWICE:**
1. Once by local-brain-client.js's own DOMContentLoaded listener
2. Once by self-dev.html's DOMContentLoaded listener

This is **not a problem** - the second call just re-confirms the URL.

---

## 3. Script Load Order in self-dev.html

```
Line 10223: <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
Line 10226: <script src="js/config.js"></script>           ← CONFIG.BRAIN_SERVER_URL set HERE
Line 10227: <script src="js/goal-system.js"></script>
Line 10228: <script src="js/local-brain-client.js"></script> ← LocalBrain.serverUrl set HERE
...
Line 10265: <script src="js/reflection-daemon.js"></script>
Line 10266: <script src="js/reflection-ui.js"></script>
Line 10269: <script src="js/map-maintenance-daemon.js"></script>
Line 10271: <script type="module" src="js/app-module.js?v=11"></script>
```

**Load Order is CORRECT:**
1. config.js loads first → reads `?brain=` param → sets `CONFIG.BRAIN_SERVER_URL`
2. local-brain-client.js loads second → reads `CONFIG.BRAIN_SERVER_URL` → sets `serverUrl`
3. No async/defer attributes - scripts execute synchronously in order

---

## 4. Hardcoded URL Search Results

### Searched Patterns:
- `trycloudflare.com` - Found only in documentation (WORKFLOW.md)
- `maple-leisure` - **NOT FOUND ANYWHERE**
- `localhost:8420` - Found as fallback default in multiple files

### Hardcoded localhost:8420 occurrences (NOT using CONFIG):
| File | Line | Usage |
|------|------|-------|
| `js/app-module.js` | 20728 | `fetch('http://localhost:8420/brain/reject-connection'` |
| `js/app-module.js` | 27979 | `fetch('http://localhost:8420/brain/learn-connection'` |

**These are BUGS** - they bypass CONFIG and LocalBrain, always hitting localhost.

### Dynamic URL usage (CORRECT):
Most code correctly uses:
```javascript
const brainUrl = window.MYND_BRAIN_URL || 'http://localhost:8420';
```
or
```javascript
LocalBrain.serverUrl
```

---

## 5. localStorage Investigation

### Keys searched:
- No localStorage key stores the brain server URL
- No caching of the server URL found

### localStorage keys used:
- `mynd-v6c` - main data storage
- `mynd-onboarded-v17` - onboarding flag
- `mynd-theme-v8` - theme preference
- `mynd-api-key` - Claude API key
- `mynd-chat-history` - chat backup
- Various neural network data keys

**Finding: The old URL is NOT stored in localStorage.**

---

## 6. Service Worker Cache Investigation (sw.js)

```javascript
const CACHE_NAME = 'mynd-v3-cache-v11';

const ASSETS_TO_CACHE = [
  '/',
  '/index.html',
  '/manifest.json'
];
```

### Caching Behavior:
- Only caches GET requests
- Caches successful (200) responses from same origin (`response.type !== 'basic'` check)
- **Does NOT cache fetch API responses to external URLs**
- **Does NOT cache POST requests** (brain server uses POST)

### However:
The service worker caches `/index.html` and other pages. If you loaded the page with `?brain=oldurl`, then:
1. The HTML was cached
2. Later requests might serve from cache WITHOUT the new URL parameter

**This is a potential issue but unlikely** since the cache is for assets, not the URL parameters.

---

## 7. Complete URL Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ USER LOADS: self-dev.html?brain=https://NEW-URL.trycloudflare.com  │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. SERVICE WORKER INTERCEPT (if installed)                          │
│    - Checks cache for self-dev.html                                 │
│    - If cached, serves OLD version (might have old embedded state?) │
│    - If not cached, fetches fresh                                   │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. config.js LOADS (synchronous)                                    │
│    - IIFE executes IMMEDIATELY                                      │
│    - Reads: new URLSearchParams(window.location.search)             │
│    - Gets: brainParam = "https://NEW-URL.trycloudflare.com"         │
│    - Sets: CONFIG.BRAIN_SERVER_URL = "https://NEW-URL..."           │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. local-brain-client.js LOADS (synchronous)                        │
│    - Object creation reads CONFIG.BRAIN_SERVER_URL                  │
│    - Sets: LocalBrain.serverUrl = "https://NEW-URL..."              │
│    - Registers DOMContentLoaded listener                            │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. DOMContentLoaded FIRES                                           │
│    - LocalBrain.init() called                                       │
│    - Checks window.MYND_BRAIN_URL (undefined)                       │
│    - Checks CONFIG.BRAIN_SERVER_URL (correct NEW-URL)               │
│    - serverUrl already correct, no change                           │
│    - Logs: "🧠 LocalBrain: Initializing with server: https://NEW..."│
│    - Calls checkAvailability() → hits NEW server                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. ROOT CAUSE HYPOTHESIS

### Most Likely: Browser HTTP Cache

The browser may be caching responses from the OLD Cloudflare URL. When requests are made:

1. Browser has cached response from `maple-leisure...trycloudflare.com`
2. New code tries to hit the same endpoint pattern
3. Browser serves cached response from wrong server

**Evidence supporting this:**
- No hardcoded old URLs in code
- URL parameter logic is correct
- Service worker doesn't cache API responses
- The "maple-leisure" URL is not in any code file

### Second Possibility: Browser Tab State

If the user kept a tab open and just refreshed:
- The page might have been loaded from browser cache (bfcache)
- JavaScript state might persist across soft refreshes

### Third Possibility: DNS/Proxy Cache

- System DNS cache holding old Cloudflare tunnel resolution
- Corporate proxy caching responses

---

## 9. Specific Fix Recommendations

### Immediate Debugging Steps:

1. **Check Browser Console on Load:**
   ```
   Look for: "🧠 LocalBrain: Initializing with server: XXX"
   ```
   This shows what URL is actually being used.

2. **Force Hard Refresh:**
   - Chrome: Cmd+Shift+R (Mac) or Ctrl+Shift+R (Win)
   - Or open DevTools → Network tab → check "Disable cache" → refresh

3. **Clear Service Worker:**
   ```javascript
   // Run in console
   navigator.serviceWorker.getRegistrations().then(regs =>
       regs.forEach(r => r.unregister())
   );
   ```

4. **Clear All Site Data:**
   - Chrome DevTools → Application → Storage → Clear site data

### Code Fixes Needed:

**Fix 1: Hardcoded localhost URLs in app-module.js**

Line 20728:
```javascript
// BEFORE (bug):
fetch('http://localhost:8420/brain/reject-connection', {

// AFTER (fix):
const brainUrl = window.MYND_BRAIN_URL || CONFIG?.BRAIN_SERVER_URL || 'http://localhost:8420';
fetch(`${brainUrl}/brain/reject-connection`, {
```

Line 27979:
```javascript
// BEFORE (bug):
fetch('http://localhost:8420/brain/learn-connection', {

// AFTER (fix):
const brainUrl = window.MYND_BRAIN_URL || CONFIG?.BRAIN_SERVER_URL || 'http://localhost:8420';
fetch(`${brainUrl}/brain/learn-connection`, {
```

**Fix 2: Add debug logging**

In config.js after BRAIN_SERVER_URL:
```javascript
console.log(`🔧 CONFIG.BRAIN_SERVER_URL set to: ${CONFIG.BRAIN_SERVER_URL}`);
console.log(`🔧 URL params:`, window.location.search);
```

**Fix 3: Service Worker versioning**

Bump `CACHE_NAME` in sw.js when deploying:
```javascript
const CACHE_NAME = 'mynd-v3-cache-v12';  // Was v11
```

---

## 10. Summary Table

| Investigation Area | Finding | Issue? |
|-------------------|---------|--------|
| Hardcoded "maple-leisure" | NOT FOUND | No |
| URL param reading | Correct timing | No |
| Script load order | Correct | No |
| LocalBrain.init() | Uses CONFIG correctly | No |
| localStorage URL caching | None | No |
| Service Worker | Caches assets, not API | Unlikely |
| Hardcoded localhost:8420 | 2 places in app-module.js | **YES - Bug** |
| Browser cache | Likely cause | **Probable Root Cause** |

---

## Conclusion

The code correctly handles the `?brain=` parameter. The old URL is **not hardcoded anywhere**. The most likely cause is **browser caching** (HTTP cache or bfcache).

**To verify:** Check the browser console immediately after load for the "LocalBrain: Initializing with server:" message. If it shows the CORRECT URL but connections still go to the old URL, then browser-level HTTP caching is the issue.

**Recommended immediate action:**
1. Hard refresh with cache disabled
2. Clear service worker
3. Clear site data
4. Fix the 2 hardcoded localhost URLs in app-module.js
