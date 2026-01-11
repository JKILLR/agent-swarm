# Models

This directory will contain SwiftData model definitions.

## Pending Models

The architect is creating corrected model definitions for:

1. **Thought.swift** - Core thought entity with:
   - `id: UUID`
   - `content: String`
   - `summary: String?`
   - `category: String?`
   - `embedding: [Float]?`
   - `position: CGPoint?`
   - `createdAt: Date`
   - `updatedAt: Date`
   - `sourceType: String`
   - `isArchived: Bool`

2. **Cluster.swift** - Grouping entity with:
   - `id: UUID`
   - `name: String`
   - `summary: String?`
   - `color: String`
   - `createdAt: Date`
   - Relationship to Thoughts

3. **Connection.swift** - Relationship entity with:
   - `id: UUID`
   - `sourceThoughtId: UUID`
   - `targetThoughtId: UUID`
   - `connectionType: String`
   - `strength: Float`
   - `isUserCreated: Bool`
   - `createdAt: Date`

## Temporary Solution

Until the models are added, `MindMapViewModel.swift` uses a temporary
`ThoughtData` class to hold thought information in memory.

Once the SwiftData models are ready:
1. Add the model files to this directory
2. Update `MYNDApp.swift` to include models in the schema
3. Update `DataService.swift` to use the actual models
4. Update `MindMapViewModel.swift` to use SwiftData queries
