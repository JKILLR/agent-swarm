# AI Corp - UI Component Architecture Review

## Executive Summary

This document provides a comprehensive analysis of React/TypeScript components needed to implement the AI Corp UI. The architecture follows a hierarchical corporate structure with real-time updates, persistent workflows, and quality gates.

---

## Component Hierarchy Tree

```
App
├── Layout
│   ├── AppShell
│   │   ├── TopNav
│   │   │   ├── Logo
│   │   │   ├── GlobalSearch
│   │   │   ├── NotificationBell
│   │   │   └── UserMenu
│   │   ├── Sidebar
│   │   │   ├── NavSection
│   │   │   ├── DepartmentNav
│   │   │   └── QuickActions
│   │   └── MainContent
│   └── PageLayout
│
├── Organization
│   ├── OrgChart (Visualization)
│   │   ├── OrgNode
│   │   ├── OrgConnection
│   │   └── OrgLegend
│   ├── DepartmentCard
│   ├── RoleCard
│   └── HierarchyBreadcrumb
│
├── Agents
│   ├── AgentCard
│   ├── AgentGrid
│   ├── AgentDetails
│   ├── WorkerPoolPanel
│   │   ├── PoolHeader
│   │   ├── WorkerList
│   │   └── PoolMetrics
│   └── AgentStatusIndicator
│
├── Molecules (Workflows)
│   ├── MoleculeCard
│   ├── MoleculeList
│   ├── MoleculeDetail
│   │   ├── MoleculeHeader
│   │   ├── StepTimeline
│   │   ├── StepCard
│   │   └── CheckpointLog
│   ├── MoleculeForm
│   └── MoleculeStatusBadge
│
├── Hooks (Work Queues)
│   ├── HookPanel
│   ├── HookQueueList
│   ├── TaskQueueItem
│   └── HookAssignmentForm
│
├── Pipeline
│   ├── PipelineBoard (Kanban)
│   │   ├── PipelineStage
│   │   ├── PipelineCard
│   │   └── PipelineStageDrop
│   ├── GateIndicator
│   ├── GateApprovalModal
│   └── PipelineProgress
│
├── Communication
│   ├── ChannelPanel
│   │   ├── UpchainMessages
│   │   ├── DownchainMessages
│   │   ├── PeerMessages
│   │   └── BroadcastMessages
│   ├── MessageCard
│   ├── MessageComposer
│   └── ChannelFilter
│
├── RACI
│   ├── RaciMatrix
│   ├── RaciCell
│   ├── RaciAssignmentForm
│   └── RaciLegend
│
├── Tasks
│   ├── TaskCard
│   ├── TaskList
│   ├── TaskDetail
│   ├── TaskForm
│   └── TaskStatusBadge
│
├── Beads (Ledger)
│   ├── BeadTimeline
│   ├── BeadEntry
│   ├── BeadFilter
│   └── BeadDiff
│
└── Shared
    ├── StatusIndicator
    ├── PriorityBadge
    ├── TimeAgo
    ├── Avatar
    ├── SkillTag
    ├── LoadingSpinner
    ├── EmptyState
    ├── ErrorBoundary
    ├── ConfirmModal
    ├── Toast
    └── Tooltip
```

---

## Complete Component Inventory

### 1. Layout Components

#### AppShell
The main application wrapper providing consistent layout structure.

```typescript
interface AppShellProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  topNav?: React.ReactNode;
}
```

#### TopNav
Global navigation header with search, notifications, and user menu.

```typescript
interface TopNavProps {
  onSearch: (query: string) => void;
  notifications: Notification[];
  user: User;
  onNotificationClick: (id: string) => void;
}

interface Notification {
  id: string;
  type: 'message' | 'gate' | 'task' | 'alert';
  title: string;
  body: string;
  read: boolean;
  timestamp: string;
  actionUrl?: string;
}
```

#### Sidebar
Navigation sidebar with department/section links.

```typescript
interface SidebarProps {
  sections: NavSection[];
  activePath: string;
  collapsed?: boolean;
  onToggleCollapse: () => void;
}

interface NavSection {
  id: string;
  label: string;
  icon: React.ReactNode;
  items: NavItem[];
}

interface NavItem {
  id: string;
  label: string;
  path: string;
  badge?: number | string;
  icon?: React.ReactNode;
}
```

---

### 2. Organization Components

#### OrgChart
Interactive organizational hierarchy visualization.

```typescript
interface OrgChartProps {
  hierarchy: OrgNode;
  onNodeClick: (nodeId: string) => void;
  selectedNodeId?: string;
  viewMode: 'full' | 'department' | 'compact';
  highlightPath?: string[]; // IDs of nodes to highlight
}

interface OrgNode {
  id: string;
  role: Role;
  agent?: Agent;
  department: Department;
  children: OrgNode[];
  status: 'active' | 'idle' | 'offline';
}
```

#### OrgNodeCard
Individual node in org chart visualization.

```typescript
interface OrgNodeCardProps {
  node: OrgNode;
  isSelected: boolean;
  onClick: () => void;
  size: 'sm' | 'md' | 'lg';
  showMetrics?: boolean;
}
```

#### DepartmentCard
Department overview card with quick stats.

```typescript
interface DepartmentCardProps {
  department: Department;
  onClick: () => void;
  metrics: DepartmentMetrics;
}

interface Department {
  id: string;
  name: 'engineering' | 'research' | 'product' | 'quality' | 'operations';
  vp: Agent;
  pools: WorkerPool[];
  directors: Agent[];
}

interface DepartmentMetrics {
  activeWorkers: number;
  totalWorkers: number;
  activeMolecules: number;
  completedToday: number;
  blockedTasks: number;
}
```

#### RoleCard
Individual role display with agent assignment.

```typescript
interface RoleCardProps {
  role: Role;
  agent?: Agent;
  onAssign?: () => void;
  showSkills?: boolean;
}

interface Role {
  id: string;
  title: string;
  level: 'vp' | 'director' | 'worker';
  department: string;
  responsibilities: string[];
  skills: string[];
  capabilities: string[];
}
```

---

### 3. Agent Components

#### AgentCard
Compact agent display for lists and grids.

```typescript
interface AgentCardProps {
  agent: Agent;
  onClick: () => void;
  showStatus?: boolean;
  showCurrentTask?: boolean;
  variant: 'compact' | 'full';
}

interface Agent {
  id: string;
  name: string;
  role: Role;
  department: string;
  status: AgentStatus;
  currentTask?: Task;
  skills: string[];
  capabilities: string[];
  metrics: AgentMetrics;
}

type AgentStatus = 'idle' | 'working' | 'blocked' | 'offline' | 'error';

interface AgentMetrics {
  tasksCompleted: number;
  averageTaskTime: number;
  successRate: number;
  currentStreak: number;
}
```

#### AgentGrid
Grid layout for displaying multiple agents.

```typescript
interface AgentGridProps {
  agents: Agent[];
  onAgentClick: (agentId: string) => void;
  filterStatus?: AgentStatus[];
  filterDepartment?: string;
  sortBy: 'name' | 'status' | 'department' | 'activity';
  viewMode: 'grid' | 'list';
}
```

#### AgentDetails
Full agent detail view with activity log.

```typescript
interface AgentDetailsProps {
  agentId: string;
  onClose: () => void;
  onAssignTask: (agentId: string) => void;
}
```

#### WorkerPoolPanel
Worker pool management and monitoring.

```typescript
interface WorkerPoolPanelProps {
  pool: WorkerPool;
  onScaleUp: () => void;
  onScaleDown: () => void;
  onWorkerClick: (workerId: string) => void;
}

interface WorkerPool {
  id: string;
  name: string;
  department: string;
  workers: Agent[];
  minWorkers: number;
  maxWorkers: number;
  skills: string[];
  capabilities: string[];
  queuedTasks: number;
}
```

#### AgentStatusIndicator
Visual status indicator (shared component).

```typescript
interface AgentStatusIndicatorProps {
  status: AgentStatus;
  size?: 'xs' | 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  pulse?: boolean; // Animate for active states
}
```

---

### 4. Molecule (Workflow) Components

#### MoleculeCard
Compact workflow card for lists.

```typescript
interface MoleculeCardProps {
  molecule: Molecule;
  onClick: () => void;
  showProgress?: boolean;
  showAccountable?: boolean;
}

interface Molecule {
  id: string;
  name: string;
  description?: string;
  status: MoleculeStatus;
  createdBy: string;
  accountable: string;
  currentStage: PipelineStage;
  steps: MoleculeStep[];
  createdAt: string;
  updatedAt: string;
  priority: Priority;
  tags?: string[];
}

type MoleculeStatus =
  | 'draft'
  | 'active'
  | 'blocked'
  | 'completed'
  | 'archived';

type PipelineStage =
  | 'inbox'
  | 'research'
  | 'design'
  | 'build'
  | 'qa'
  | 'security'
  | 'deploy'
  | 'monitor'
  | 'complete'
  | 'archive';

type Priority = 'critical' | 'high' | 'medium' | 'low';
```

#### MoleculeList
Filterable list of molecules.

```typescript
interface MoleculeListProps {
  molecules: Molecule[];
  onMoleculeClick: (id: string) => void;
  onCreateNew: () => void;
  filters: MoleculeFilters;
  onFiltersChange: (filters: MoleculeFilters) => void;
}

interface MoleculeFilters {
  status?: MoleculeStatus[];
  stage?: PipelineStage[];
  department?: string[];
  accountable?: string;
  priority?: Priority[];
  search?: string;
}
```

#### MoleculeDetail
Full molecule view with steps and checkpoints.

```typescript
interface MoleculeDetailProps {
  moleculeId: string;
  onClose: () => void;
  onEdit: () => void;
  onAdvanceStage: () => void;
}
```

#### MoleculeForm
Form for creating/editing molecules.

```typescript
interface MoleculeFormProps {
  molecule?: Molecule; // Undefined for create
  onSubmit: (data: MoleculeFormData) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

interface MoleculeFormData {
  name: string;
  description?: string;
  accountable: string;
  priority: Priority;
  steps: MoleculeStepFormData[];
  tags?: string[];
}

interface MoleculeStepFormData {
  name: string;
  assignedTo: string;
  dependsOn?: string[];
  isGate?: boolean;
}
```

#### StepTimeline
Visual timeline of molecule steps.

```typescript
interface StepTimelineProps {
  steps: MoleculeStep[];
  currentStepId?: string;
  onStepClick: (stepId: string) => void;
  orientation: 'horizontal' | 'vertical';
}

interface MoleculeStep {
  id: string;
  name: string;
  status: StepStatus;
  assignedTo: string;
  completedBy?: string;
  completedAt?: string;
  checkpoint?: string;
  dependsOn: string[];
  isGate: boolean;
}

type StepStatus = 'pending' | 'in_progress' | 'completed' | 'blocked' | 'failed';
```

#### StepCard
Individual step details card.

```typescript
interface StepCardProps {
  step: MoleculeStep;
  isActive: boolean;
  onClick: () => void;
  onUpdateCheckpoint: (checkpoint: string) => void;
}
```

#### MoleculeStatusBadge
Visual status badge for molecules.

```typescript
interface MoleculeStatusBadgeProps {
  status: MoleculeStatus;
  stage?: PipelineStage;
  size?: 'sm' | 'md';
}
```

---

### 5. Hook (Work Queue) Components

#### HookPanel
Agent's work queue panel.

```typescript
interface HookPanelProps {
  agentId: string;
  hook: Hook;
  onClaimTask: (taskId: string) => void;
  onCompleteTask: (taskId: string) => void;
}

interface Hook {
  agentId: string;
  department: string;
  role: string;
  currentTask?: Task;
  queue: QueuedTask[];
}

interface QueuedTask {
  taskId: string;
  moleculeId: string;
  priority: Priority;
  addedAt: string;
  estimatedDuration?: number;
}
```

#### HookQueueList
List of queued tasks for an agent.

```typescript
interface HookQueueListProps {
  queue: QueuedTask[];
  onTaskClick: (taskId: string) => void;
  onReorder: (taskIds: string[]) => void;
  draggable?: boolean;
}
```

#### TaskQueueItem
Individual task in queue.

```typescript
interface TaskQueueItemProps {
  task: QueuedTask;
  position: number;
  onClick: () => void;
  onClaim?: () => void;
  isDragging?: boolean;
}
```

#### HookAssignmentForm
Form to assign tasks to hooks.

```typescript
interface HookAssignmentFormProps {
  task: Task;
  availableAgents: Agent[];
  onAssign: (agentId: string, priority: Priority) => void;
  onCancel: () => void;
}
```

---

### 6. Pipeline Components

#### PipelineBoard
Kanban-style pipeline visualization.

```typescript
interface PipelineBoardProps {
  molecules: Molecule[];
  stages: PipelineStageConfig[];
  onMoleculeMove: (moleculeId: string, newStage: PipelineStage) => void;
  onMoleculeClick: (moleculeId: string) => void;
  onGateApproval: (moleculeId: string, gateId: string) => void;
}

interface PipelineStageConfig {
  id: PipelineStage;
  label: string;
  color: string;
  icon: React.ReactNode;
  hasGate: boolean;
  gateLabel?: string;
}
```

#### PipelineStage
Individual stage column in kanban.

```typescript
interface PipelineStageProps {
  stage: PipelineStageConfig;
  molecules: Molecule[];
  onDrop: (moleculeId: string) => void;
  isDropTarget: boolean;
  onMoleculeClick: (moleculeId: string) => void;
}
```

#### PipelineCard
Draggable molecule card in pipeline.

```typescript
interface PipelineCardProps {
  molecule: Molecule;
  onClick: () => void;
  isDragging: boolean;
  blockedByGate: boolean;
}
```

#### GateIndicator
Quality gate status indicator.

```typescript
interface GateIndicatorProps {
  gate: Gate;
  onClick: () => void;
  size?: 'sm' | 'md' | 'lg';
}

interface Gate {
  id: string;
  stage: PipelineStage;
  status: 'pending' | 'approved' | 'rejected';
  approvedBy?: string;
  approvedAt?: string;
  rejectionReason?: string;
  requirements: GateRequirement[];
}

interface GateRequirement {
  id: string;
  label: string;
  satisfied: boolean;
  evidence?: string;
}
```

#### GateApprovalModal
Modal for approving/rejecting gates.

```typescript
interface GateApprovalModalProps {
  gate: Gate;
  molecule: Molecule;
  onApprove: (notes?: string) => void;
  onReject: (reason: string) => void;
  onClose: () => void;
  canApprove: boolean;
}
```

#### PipelineProgress
Linear progress indicator across stages.

```typescript
interface PipelineProgressProps {
  currentStage: PipelineStage;
  completedStages: PipelineStage[];
  blockedAt?: PipelineStage;
  compact?: boolean;
}
```

---

### 7. Communication Components

#### ChannelPanel
Communication channel viewer.

```typescript
interface ChannelPanelProps {
  channelType: ChannelType;
  messages: Message[];
  onSendMessage: (message: MessageDraft) => void;
  onMessageClick: (messageId: string) => void;
  currentAgentId: string;
}

type ChannelType = 'upchain' | 'downchain' | 'peer' | 'broadcast';

interface Message {
  id: string;
  channelType: ChannelType;
  from: string;
  to: string | string[];
  subject: string;
  body: string;
  timestamp: string;
  read: boolean;
  priority: Priority;
  relatedMoleculeId?: string;
  relatedTaskId?: string;
  threadId?: string;
}

interface MessageDraft {
  to: string | string[];
  subject: string;
  body: string;
  priority: Priority;
  relatedMoleculeId?: string;
}
```

#### MessageCard
Individual message display.

```typescript
interface MessageCardProps {
  message: Message;
  onClick: () => void;
  showThread?: boolean;
  compact?: boolean;
}
```

#### MessageComposer
Message composition form.

```typescript
interface MessageComposerProps {
  channelType: ChannelType;
  recipientOptions: Agent[];
  onSend: (message: MessageDraft) => void;
  onCancel: () => void;
  replyTo?: Message;
  defaultRecipient?: string;
}
```

#### ChannelFilter
Filter controls for messages.

```typescript
interface ChannelFilterProps {
  filters: MessageFilters;
  onChange: (filters: MessageFilters) => void;
}

interface MessageFilters {
  channelTypes: ChannelType[];
  unreadOnly: boolean;
  from?: string[];
  priority?: Priority[];
  dateRange?: { start: string; end: string };
}
```

---

### 8. RACI Components

#### RaciMatrix
RACI responsibility matrix visualization.

```typescript
interface RaciMatrixProps {
  molecule: Molecule;
  assignments: RaciAssignment[];
  agents: Agent[];
  onAssignmentChange: (assignment: RaciAssignment) => void;
  readOnly?: boolean;
}

interface RaciAssignment {
  stepId: string;
  agentId: string;
  role: RaciRole;
}

type RaciRole = 'responsible' | 'accountable' | 'consulted' | 'informed';
```

#### RaciCell
Individual cell in RACI matrix.

```typescript
interface RaciCellProps {
  stepId: string;
  agentId: string;
  role?: RaciRole;
  onChange: (role: RaciRole | null) => void;
  disabled?: boolean;
  isAccountableLocked?: boolean;
}
```

#### RaciAssignmentForm
Form for RACI assignments.

```typescript
interface RaciAssignmentFormProps {
  step: MoleculeStep;
  availableAgents: Agent[];
  currentAssignments: RaciAssignment[];
  onSubmit: (assignments: RaciAssignment[]) => void;
  onCancel: () => void;
}
```

#### RaciLegend
Legend explaining RACI roles.

```typescript
interface RaciLegendProps {
  compact?: boolean;
}
```

---

### 9. Task Components

#### TaskCard
Compact task display.

```typescript
interface TaskCardProps {
  task: Task;
  onClick: () => void;
  showAssignee?: boolean;
  showMolecule?: boolean;
  draggable?: boolean;
}

interface Task {
  id: string;
  title: string;
  description?: string;
  status: TaskStatus;
  priority: Priority;
  assignedTo?: string;
  moleculeId: string;
  stepId: string;
  createdAt: string;
  updatedAt: string;
  dueDate?: string;
  checkpoint?: string;
  blockers?: string[];
  tags?: string[];
}

type TaskStatus = 'pending' | 'claimed' | 'in_progress' | 'blocked' | 'completed' | 'failed';
```

#### TaskList
Filterable task list.

```typescript
interface TaskListProps {
  tasks: Task[];
  onTaskClick: (taskId: string) => void;
  onTaskStatusChange: (taskId: string, status: TaskStatus) => void;
  filters: TaskFilters;
  onFiltersChange: (filters: TaskFilters) => void;
  groupBy?: 'status' | 'priority' | 'assignee' | 'molecule';
}

interface TaskFilters {
  status?: TaskStatus[];
  priority?: Priority[];
  assignedTo?: string[];
  moleculeId?: string;
  search?: string;
}
```

#### TaskDetail
Full task view with activity.

```typescript
interface TaskDetailProps {
  taskId: string;
  onClose: () => void;
  onEdit: () => void;
  onStatusChange: (status: TaskStatus) => void;
  onAddBlocker: (blocker: string) => void;
}
```

#### TaskForm
Form for creating/editing tasks.

```typescript
interface TaskFormProps {
  task?: Task;
  molecule: Molecule;
  step: MoleculeStep;
  onSubmit: (data: TaskFormData) => void;
  onCancel: () => void;
  isLoading?: boolean;
}

interface TaskFormData {
  title: string;
  description?: string;
  priority: Priority;
  assignedTo?: string;
  dueDate?: string;
  tags?: string[];
}
```

#### TaskStatusBadge
Visual task status indicator.

```typescript
interface TaskStatusBadgeProps {
  status: TaskStatus;
  size?: 'sm' | 'md';
  showIcon?: boolean;
}
```

---

### 10. Bead (Ledger) Components

#### BeadTimeline
Git-backed audit trail timeline.

```typescript
interface BeadTimelineProps {
  beads: Bead[];
  onBeadClick: (beadId: string) => void;
  filters: BeadFilters;
  onFiltersChange: (filters: BeadFilters) => void;
}

interface Bead {
  id: string;
  type: BeadType;
  action: string;
  actor: string;
  timestamp: string;
  moleculeId?: string;
  taskId?: string;
  data: Record<string, unknown>;
  gitCommit: string;
  parentBeadId?: string;
}

type BeadType = 'task' | 'decision' | 'handoff' | 'gate' | 'message' | 'checkpoint';
```

#### BeadEntry
Individual audit log entry.

```typescript
interface BeadEntryProps {
  bead: Bead;
  onClick: () => void;
  showDiff?: boolean;
  expanded?: boolean;
}
```

#### BeadFilter
Audit log filters.

```typescript
interface BeadFilterProps {
  filters: BeadFilters;
  onChange: (filters: BeadFilters) => void;
}

interface BeadFilters {
  types?: BeadType[];
  actors?: string[];
  moleculeId?: string;
  dateRange?: { start: string; end: string };
}
```

#### BeadDiff
Git diff visualization for bead.

```typescript
interface BeadDiffProps {
  bead: Bead;
  previousBead?: Bead;
  showRaw?: boolean;
}
```

---

### 11. Shared Components

#### StatusIndicator
Generic status dot/badge.

```typescript
interface StatusIndicatorProps {
  status: 'success' | 'warning' | 'error' | 'info' | 'neutral' | 'active';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  pulse?: boolean;
  label?: string;
}
```

#### PriorityBadge
Priority level badge.

```typescript
interface PriorityBadgeProps {
  priority: Priority;
  size?: 'sm' | 'md';
  showIcon?: boolean;
}
```

#### TimeAgo
Relative time display.

```typescript
interface TimeAgoProps {
  timestamp: string;
  showAbsolute?: boolean;
  updateInterval?: number;
}
```

#### Avatar
Agent/user avatar.

```typescript
interface AvatarProps {
  agent?: Agent;
  name?: string;
  imageUrl?: string;
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  showStatus?: boolean;
  status?: AgentStatus;
}
```

#### SkillTag
Skill capability tag.

```typescript
interface SkillTagProps {
  skill: string;
  variant?: 'default' | 'primary' | 'secondary';
  removable?: boolean;
  onRemove?: () => void;
}
```

#### LoadingSpinner
Loading state indicator.

```typescript
interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  fullScreen?: boolean;
}
```

#### EmptyState
Empty state placeholder.

```typescript
interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}
```

#### ErrorBoundary
Error handling wrapper.

```typescript
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}
```

#### ConfirmModal
Confirmation dialog.

```typescript
interface ConfirmModalProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'default' | 'danger';
  onConfirm: () => void;
  onCancel: () => void;
  isLoading?: boolean;
}
```

#### Toast
Toast notification.

```typescript
interface ToastProps {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  description?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}
```

#### Tooltip
Hover tooltip.

```typescript
interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  position?: 'top' | 'right' | 'bottom' | 'left';
  delay?: number;
}
```

---

## Component Categories

### Shared vs Specialized Components

#### Shared Components (Reusable Across Features)
| Component | Usage |
|-----------|-------|
| `StatusIndicator` | Agent status, task status, molecule status, gate status |
| `PriorityBadge` | Tasks, molecules, messages |
| `TimeAgo` | All timestamps throughout the app |
| `Avatar` | Agent displays, message threads, activity logs |
| `SkillTag` | Agent details, pool configs, role displays |
| `LoadingSpinner` | All async operations |
| `EmptyState` | All list/grid views when empty |
| `ErrorBoundary` | All component trees |
| `ConfirmModal` | Destructive actions, gate approvals |
| `Toast` | All user feedback |
| `Tooltip` | Throughout for additional context |

#### Specialized Components (Feature-Specific)
| Feature | Components |
|---------|------------|
| Organization | `OrgChart`, `OrgNodeCard`, `DepartmentCard`, `RoleCard` |
| Pipeline | `PipelineBoard`, `PipelineStage`, `GateIndicator`, `GateApprovalModal` |
| Molecules | `MoleculeForm`, `StepTimeline`, `MoleculeDetail` |
| Communication | `ChannelPanel`, `MessageComposer`, `ChannelFilter` |
| RACI | `RaciMatrix`, `RaciCell`, `RaciAssignmentForm` |
| Beads | `BeadTimeline`, `BeadDiff`, `BeadEntry` |

---

## State Management Considerations

### Global State (Context/Redux)

```typescript
interface AppState {
  // User/Session
  currentUser: User;
  currentAgentContext?: Agent; // For impersonation/viewing as agent

  // Organization
  orgHierarchy: OrgNode;
  departments: Department[];
  agents: Agent[];
  workerPools: WorkerPool[];

  // Work Items
  molecules: Molecule[];
  tasks: Task[];
  hooks: Hook[];

  // Communication
  messages: Message[];
  unreadCount: number;

  // Pipeline
  pipelineConfig: PipelineStageConfig[];
  gates: Gate[];

  // Audit
  recentBeads: Bead[];

  // UI State
  selectedMoleculeId?: string;
  selectedAgentId?: string;
  sidebarCollapsed: boolean;
  activeFilters: Record<string, unknown>;

  // WebSocket
  connectionStatus: 'connected' | 'connecting' | 'disconnected';
  lastSyncTimestamp: string;
}
```

### State Slices

```typescript
// Suggested Redux Toolkit slices or React Context organization

// 1. Organization Slice
interface OrgState {
  hierarchy: OrgNode;
  departments: Department[];
  agents: Record<string, Agent>;
  pools: Record<string, WorkerPool>;
  loading: boolean;
  error?: string;
}

// 2. Molecules Slice
interface MoleculesState {
  items: Record<string, Molecule>;
  activeIds: string[];
  filters: MoleculeFilters;
  selectedId?: string;
  loading: boolean;
}

// 3. Tasks Slice
interface TasksState {
  items: Record<string, Task>;
  byMolecule: Record<string, string[]>;
  byAgent: Record<string, string[]>;
  filters: TaskFilters;
  loading: boolean;
}

// 4. Communication Slice
interface CommunicationState {
  messages: Record<string, Message>;
  byChannel: Record<ChannelType, string[]>;
  threads: Record<string, string[]>;
  unreadCount: number;
  filters: MessageFilters;
}

// 5. Pipeline Slice
interface PipelineState {
  stages: PipelineStageConfig[];
  gates: Record<string, Gate>;
  moleculesByStage: Record<PipelineStage, string[]>;
}

// 6. UI Slice
interface UIState {
  sidebarCollapsed: boolean;
  modals: Record<string, boolean>;
  toasts: Toast[];
  theme: 'light' | 'dark';
}
```

### Local Component State

Components should manage local state for:
- Form inputs before submission
- UI interactions (hover, focus, expand/collapse)
- Transient loading states
- Drag-and-drop operations
- Optimistic updates before confirmation

---

## WebSocket Real-Time Update Patterns

### Connection Management

```typescript
interface WebSocketConfig {
  url: string;
  reconnectAttempts: number;
  reconnectInterval: number;
  heartbeatInterval: number;
}

interface WebSocketMessage {
  type: WebSocketEventType;
  payload: unknown;
  timestamp: string;
  correlationId?: string;
}

type WebSocketEventType =
  // Agent Events
  | 'agent.status_changed'
  | 'agent.task_claimed'
  | 'agent.task_completed'
  | 'agent.blocked'

  // Molecule Events
  | 'molecule.created'
  | 'molecule.updated'
  | 'molecule.stage_changed'
  | 'molecule.completed'

  // Task Events
  | 'task.created'
  | 'task.assigned'
  | 'task.status_changed'
  | 'task.checkpoint_updated'

  // Gate Events
  | 'gate.pending'
  | 'gate.approved'
  | 'gate.rejected'

  // Message Events
  | 'message.received'
  | 'message.read'

  // Pool Events
  | 'pool.worker_added'
  | 'pool.worker_removed'
  | 'pool.scaled'

  // System Events
  | 'system.sync_required'
  | 'system.maintenance';
```

### WebSocket Hook

```typescript
// Custom hook for WebSocket subscription
function useWebSocket(config: WebSocketConfig) {
  const [status, setStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  // Connection management
  const connect = useCallback(() => { /* ... */ }, []);
  const disconnect = useCallback(() => { /* ... */ }, []);
  const send = useCallback((message: WebSocketMessage) => { /* ... */ }, []);

  // Subscribe to specific event types
  const subscribe = useCallback((
    eventType: WebSocketEventType,
    handler: (payload: unknown) => void
  ) => { /* ... */ }, []);

  return { status, lastMessage, connect, disconnect, send, subscribe };
}
```

### Event-Specific Hooks

```typescript
// Hook for agent status updates
function useAgentStatusUpdates(agentId: string) {
  const { subscribe } = useWebSocket(config);
  const dispatch = useDispatch();

  useEffect(() => {
    const unsubscribe = subscribe('agent.status_changed', (payload) => {
      if (payload.agentId === agentId) {
        dispatch(updateAgentStatus(payload));
      }
    });
    return unsubscribe;
  }, [agentId, subscribe, dispatch]);
}

// Hook for molecule updates
function useMoleculeUpdates(moleculeId?: string) {
  const { subscribe } = useWebSocket(config);
  const dispatch = useDispatch();

  useEffect(() => {
    const events: WebSocketEventType[] = [
      'molecule.updated',
      'molecule.stage_changed',
      'molecule.completed'
    ];

    const unsubscribes = events.map(event =>
      subscribe(event, (payload) => {
        if (!moleculeId || payload.moleculeId === moleculeId) {
          dispatch(handleMoleculeEvent(event, payload));
        }
      })
    );

    return () => unsubscribes.forEach(unsub => unsub());
  }, [moleculeId, subscribe, dispatch]);
}

// Hook for pipeline board real-time updates
function usePipelineBoardUpdates() {
  const { subscribe } = useWebSocket(config);
  const dispatch = useDispatch();

  useEffect(() => {
    const events: WebSocketEventType[] = [
      'molecule.stage_changed',
      'gate.approved',
      'gate.rejected',
      'gate.pending'
    ];

    const unsubscribes = events.map(event =>
      subscribe(event, (payload) => {
        dispatch(handlePipelineEvent(event, payload));
      })
    );

    return () => unsubscribes.forEach(unsub => unsub());
  }, [subscribe, dispatch]);
}
```

### Optimistic Updates Pattern

```typescript
// Pattern for optimistic updates with rollback
async function optimisticUpdate<T>(
  dispatch: Dispatch,
  optimisticAction: Action,
  apiCall: () => Promise<T>,
  rollbackAction: Action,
  successAction: (result: T) => Action
) {
  // Apply optimistic update
  dispatch(optimisticAction);

  try {
    const result = await apiCall();
    dispatch(successAction(result));
  } catch (error) {
    // Rollback on failure
    dispatch(rollbackAction);
    throw error;
  }
}

// Example: Moving molecule to new stage
async function moveMoleculeToStage(
  moleculeId: string,
  newStage: PipelineStage
) {
  const previousStage = selectMoleculeStage(moleculeId);

  await optimisticUpdate(
    dispatch,
    // Optimistic
    moleculeActions.setStage({ moleculeId, stage: newStage }),
    // API call
    () => api.molecules.updateStage(moleculeId, newStage),
    // Rollback
    moleculeActions.setStage({ moleculeId, stage: previousStage }),
    // Success
    (result) => moleculeActions.updateSuccess(result)
  );
}
```

---

## Form Components Summary

| Form Component | Purpose | Key Fields |
|----------------|---------|------------|
| `MoleculeForm` | Create/edit workflows | name, description, accountable, priority, steps |
| `TaskForm` | Create/edit tasks | title, description, priority, assignedTo, dueDate |
| `HookAssignmentForm` | Assign tasks to agents | agent selection, priority |
| `RaciAssignmentForm` | Set RACI responsibilities | agent assignments per role |
| `GateApprovalModal` | Approve/reject gates | approval notes, rejection reason |
| `MessageComposer` | Send messages | recipient, subject, body, priority |

---

## List/Table Components Summary

| Component | Data Type | Features |
|-----------|-----------|----------|
| `MoleculeList` | Molecules | Filter, sort, search, pagination |
| `TaskList` | Tasks | Filter, group by, status change |
| `AgentGrid` | Agents | Filter, sort, grid/list toggle |
| `HookQueueList` | Queued tasks | Drag reorder, claim action |
| `BeadTimeline` | Audit entries | Filter, expand/collapse |
| `WorkerList` | Pool workers | Status indicators, click to detail |

---

## Visualization Components Summary

| Component | Visualization Type | Library Suggestion |
|-----------|-------------------|-------------------|
| `OrgChart` | Hierarchical tree | react-flow, d3-hierarchy |
| `PipelineBoard` | Kanban board | react-beautiful-dnd, dnd-kit |
| `StepTimeline` | Step progression | Custom CSS, react-vertical-timeline |
| `BeadTimeline` | Audit timeline | Custom, react-chrono |
| `PipelineProgress` | Linear progress | Custom CSS |
| `RaciMatrix` | Grid/table | react-table, custom grid |
| `BeadDiff` | Git diff view | react-diff-viewer |

---

## Status Indicator Components Summary

| Component | Statuses | Visual Style |
|-----------|----------|--------------|
| `AgentStatusIndicator` | idle, working, blocked, offline, error | Colored dot with optional pulse |
| `MoleculeStatusBadge` | draft, active, blocked, completed, archived | Colored badge with stage icon |
| `TaskStatusBadge` | pending, claimed, in_progress, blocked, completed, failed | Icon + color badge |
| `GateIndicator` | pending, approved, rejected | Traffic light / lock icon |
| `StatusIndicator` | Generic states | Configurable dot/badge |
| `PriorityBadge` | critical, high, medium, low | Color-coded with icon |

---

## Implementation Priority Recommendations

### Phase 1: Foundation
1. Shared components (StatusIndicator, Avatar, Toast, etc.)
2. Layout components (AppShell, Sidebar, TopNav)
3. Basic agent and molecule cards/lists

### Phase 2: Core Features
1. OrgChart visualization
2. PipelineBoard (Kanban)
3. MoleculeDetail with StepTimeline
4. Task management components

### Phase 3: Advanced Features
1. Communication/Channel components
2. RACI matrix
3. Hook management
4. BeadTimeline and audit features

### Phase 4: Polish
1. Real-time WebSocket integration
2. Optimistic updates
3. Advanced filtering/search
4. Performance optimization

---

## File Organization Recommendation

```
src/
├── components/
│   ├── layout/
│   │   ├── AppShell.tsx
│   │   ├── TopNav.tsx
│   │   ├── Sidebar.tsx
│   │   └── index.ts
│   ├── organization/
│   │   ├── OrgChart/
│   │   │   ├── OrgChart.tsx
│   │   │   ├── OrgNodeCard.tsx
│   │   │   ├── OrgConnection.tsx
│   │   │   └── index.ts
│   │   ├── DepartmentCard.tsx
│   │   ├── RoleCard.tsx
│   │   └── index.ts
│   ├── agents/
│   │   ├── AgentCard.tsx
│   │   ├── AgentGrid.tsx
│   │   ├── AgentDetails.tsx
│   │   ├── WorkerPoolPanel.tsx
│   │   ├── AgentStatusIndicator.tsx
│   │   └── index.ts
│   ├── molecules/
│   │   ├── MoleculeCard.tsx
│   │   ├── MoleculeList.tsx
│   │   ├── MoleculeDetail/
│   │   ├── MoleculeForm.tsx
│   │   ├── StepTimeline.tsx
│   │   └── index.ts
│   ├── hooks/
│   │   ├── HookPanel.tsx
│   │   ├── HookQueueList.tsx
│   │   └── index.ts
│   ├── pipeline/
│   │   ├── PipelineBoard/
│   │   ├── GateIndicator.tsx
│   │   ├── GateApprovalModal.tsx
│   │   └── index.ts
│   ├── communication/
│   │   ├── ChannelPanel.tsx
│   │   ├── MessageCard.tsx
│   │   ├── MessageComposer.tsx
│   │   └── index.ts
│   ├── raci/
│   │   ├── RaciMatrix.tsx
│   │   ├── RaciCell.tsx
│   │   └── index.ts
│   ├── tasks/
│   │   ├── TaskCard.tsx
│   │   ├── TaskList.tsx
│   │   ├── TaskForm.tsx
│   │   └── index.ts
│   ├── beads/
│   │   ├── BeadTimeline.tsx
│   │   ├── BeadEntry.tsx
│   │   ├── BeadDiff.tsx
│   │   └── index.ts
│   └── shared/
│       ├── StatusIndicator.tsx
│       ├── PriorityBadge.tsx
│       ├── TimeAgo.tsx
│       ├── Avatar.tsx
│       ├── SkillTag.tsx
│       ├── LoadingSpinner.tsx
│       ├── EmptyState.tsx
│       ├── ErrorBoundary.tsx
│       ├── ConfirmModal.tsx
│       ├── Toast.tsx
│       ├── Tooltip.tsx
│       └── index.ts
├── hooks/
│   ├── useWebSocket.ts
│   ├── useAgentStatusUpdates.ts
│   ├── useMoleculeUpdates.ts
│   ├── usePipelineBoardUpdates.ts
│   └── index.ts
├── store/
│   ├── slices/
│   │   ├── orgSlice.ts
│   │   ├── moleculesSlice.ts
│   │   ├── tasksSlice.ts
│   │   ├── communicationSlice.ts
│   │   ├── pipelineSlice.ts
│   │   └── uiSlice.ts
│   └── index.ts
└── types/
    ├── agent.ts
    ├── molecule.ts
    ├── task.ts
    ├── organization.ts
    ├── pipeline.ts
    ├── communication.ts
    └── index.ts
```

---

## Conclusion

This component architecture provides a comprehensive foundation for implementing the AI Corp UI. The design emphasizes:

1. **Reusability** - Shared components reduce duplication
2. **Type Safety** - Complete TypeScript interfaces for all props
3. **Real-time Updates** - WebSocket patterns for live data
4. **Scalability** - Organized state management with slices
5. **Maintainability** - Clear component hierarchy and file organization

The architecture maps directly to the AI Corp domain concepts while maintaining flexibility for future enhancements.
