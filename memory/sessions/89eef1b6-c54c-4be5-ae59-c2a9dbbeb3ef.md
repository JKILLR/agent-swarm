# Session 89eef1b6-c54c-4be5-ae59-c2a9dbbeb3ef

**User**: great. now another fix - when work is happening, if the activity panel shows "agents working" and "files modified" at the same time the expand button is covered. this needs to be reconfigured. the fil...

**COO Response**: The fix:
1. Changed `gap-1` to `gap-2` for better spacing between elements
2. Added `flex-shrink-0` to prevent the button area from being compressed
3. Replaced the prominent yellow badge (`text-xs text-yellow-400 bg-yellow-500/10 px-2 py-0.5 rounded mr-2`) with a much more subtle indicator:
   - Small `FileEdit` icon (3x3) in muted yellow (`text-yellow-500/70`)
   - Tiny count number (`text-[10px]`)
   - Tooltip on hover shows full "X files modified" text
   - Removed the background and padding...