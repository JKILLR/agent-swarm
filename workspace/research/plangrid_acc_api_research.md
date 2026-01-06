# PlanGrid and Autodesk Construction Cloud API Research

**Date**: 2026-01-06
**Researcher**: Research Specialist Agent
**Status**: Complete

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Legacy PlanGrid API](#legacy-plangrid-api)
3. [Autodesk Platform Services (APS) Overview](#autodesk-platform-services-aps-overview)
4. [Authentication (OAuth 2.0)](#authentication-oauth-20)
5. [Key Data Objects and APIs](#key-data-objects-and-apis)
6. [BIM 360 and ACC APIs](#bim-360-and-acc-apis)
7. [Webhook Capabilities](#webhook-capabilities)
8. [Rate Limits and Access Tiers](#rate-limits-and-access-tiers)
9. [API Reference Links](#api-reference-links)

---

## Executive Summary

PlanGrid was acquired by Autodesk in November 2018 and has been integrated into Autodesk Construction Cloud (ACC). The legacy PlanGrid API at `developer.plangrid.com` remains available but is being deprecated in favor of Autodesk Platform Services (APS) APIs.

**Key Transition Notes:**
- PlanGrid is no longer available for purchase by new customers
- Autodesk Build was developed combining PlanGrid and BIM 360 features
- When migrating to Autodesk Build, all API tokens must be reset
- New development should target APS/ACC APIs

---

## Legacy PlanGrid API

### Documentation
- **Developer Portal**: https://developer.plangrid.com
- **Support Email**: plangrid-api@autodesk.com

### Available Endpoints (Legacy)
- Projects
- Sheets (drawings/plans)
- Users
- Issues
- Photos
- Field Reports
- Tasks (punch lists)

### PlanGrid Connect
- No-code integration platform (now Autodesk Construction Connect)
- Built on Workato
- Allows flexible integrations without engineering resources

### Deprecation Status
- PlanGrid is a legacy tool
- Features migrated to Autodesk Build
- API tokens reset during migration
- Recommend transitioning to APS/ACC APIs

---

## Autodesk Platform Services (APS) Overview

**Formerly Known As**: Autodesk Forge

**Official Documentation**:
- Main Portal: https://aps.autodesk.com
- ACC API Overview: https://aps.autodesk.com/en/docs/acc/v1/overview/
- API Reference: https://aps.autodesk.com/en/docs/acc/v1/reference

### Platform Architecture
```
Autodesk Platform Services (APS)
|
+-- Authentication API (OAuth 2.0)
+-- Data Management API
|   +-- Hubs
|   +-- Projects
|   +-- Folders
|   +-- Items
|   +-- Versions
|
+-- Autodesk Construction Cloud APIs
|   +-- Issues API
|   +-- RFIs API
|   +-- Submittals API
|   +-- Photos API
|   +-- Assets API
|   +-- Locations API
|   +-- Checklists API
|   +-- Cost Management API
|   +-- Daily Logs API
|   +-- Meeting Minutes API
|
+-- BIM 360 APIs
|   +-- Document Management
|   +-- Model Coordination
|   +-- Field Management
|   +-- Project Management
|
+-- Webhooks API
+-- Data Connector API
+-- Model Derivative API
+-- Viewer API
```

---

## Authentication (OAuth 2.0)

### Overview
APS uses OAuth 2.0 for all API authentication. Tokens are required for accessing any API endpoints.

### Authentication Types

#### 2-Legged Authentication (Client Credentials Grant)
- **Use Case**: Server-to-server communication, no user context needed
- **Flow**: App directly communicates with APS for token
- **Required For**: Some account-level operations (e.g., GET Account Users)

**Token Request:**
```http
POST https://developer.api.autodesk.com/authentication/v2/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_id=YOUR_CLIENT_ID
&client_secret=YOUR_CLIENT_SECRET
&scope=data:read
```

#### 3-Legged Authentication (Authorization Code Grant)
- **Use Case**: User-authorized access to their data
- **Flow**: User redirected to Autodesk login, returns with auth code, exchange for token
- **Required For**: Most ACC APIs (Issues, RFIs, etc.)
- **PKCE Support**: Available for desktop/mobile/SPA apps

**Authorization URL:**
```
https://developer.api.autodesk.com/authentication/v2/authorize
?response_type=code
&client_id=YOUR_CLIENT_ID
&redirect_uri=YOUR_CALLBACK_URL
&scope=data:read data:write
```

**Token Exchange:**
```http
POST https://developer.api.autodesk.com/authentication/v2/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&code=AUTHORIZATION_CODE
&client_id=YOUR_CLIENT_ID
&client_secret=YOUR_CLIENT_SECRET
&redirect_uri=YOUR_CALLBACK_URL
```

### Common OAuth Scopes

| Scope | Description |
|-------|-------------|
| `data:read` | Read access to data |
| `data:write` | Write access to data |
| `data:create` | Create new data objects |
| `data:search` | Search data |
| `account:read` | Read account information |
| `account:write` | Write account information |
| `user-profile:read` | Read user profile |
| `user:read` | Read user info |
| `user:write` | Write user info |
| `viewables:read` | Read viewable derivatives |
| `bucket:create` | Create OSS buckets |
| `bucket:read` | Read OSS buckets |
| `code:all` | Design Automation |

### Token Best Practices
- Provide both 2-legged and 3-legged tokens with appropriate scopes
- SDK auto-refreshes tokens as they expire
- For Issues API, only 3-legged tokens work
- Custom integrations require adding APS client ID in ACC settings

---

## Key Data Objects and APIs

### Data Management API Hierarchy

```
Hub (Account)
  |
  +-- Project
        |
        +-- Folder
              |
              +-- Item
                    |
                    +-- Version
```

### Projects API
- **Endpoint**: `/project/v1/hubs/{hub_id}/projects`
- **Auth**: 3-legged token with `data:read` scope
- Access project metadata, root folders, member permissions

### Folders API
- **Endpoint**: `/data/v1/projects/{project_id}/folders`
- Navigate folder hierarchy
- Access top-level folders via `topFolders` endpoint

### Items and Versions API
- Items represent files/documents
- Versions track revisions
- Download files via Object Storage Service (OSS)

### Sheets/Drawings
- Located under "Plans" folder in ACC
- Upload workflow: Upload -> Title block extraction (OCR) -> Sheet publishing
- View in browser via Viewer API
- 2D and 3D visualization supported

### Issues API
- **Endpoint**: `/construction/issues/v1/projects/{project_id}/issues`
- **Auth**: 3-legged token only
- **System Name (Webhooks)**: `autodesk.construction.issues`

**Issue Object Properties:**
- id, title, description
- status (draft, open, pending, closed, void)
- type (issue type from project settings)
- assignedTo (user ID)
- dueDate
- location (coordinates, sheet reference)
- attachments
- comments

### RFIs API
- **Endpoint**: `/bim360/rfis/v2/containers/{container_id}/rfis`
- Use `/bim360/` endpoints for ACC compatibility
- Link to issues, locations, attachments

### Submittals API (Generally Available)
- **Endpoint**: `/construction/submittals/v1/projects/{project_id}/items`

**Capabilities:**
- Create/read submittal items
- Manage spec sections
- Handle packages
- Track review workflows
- Manage attachments
- Custom numbering support

### Photos API
- **Endpoint**: `/construction/photos/v1/projects/{project_id}/photos`
- Document progress, materials, issues
- Link to field reports

### Daily Reports / Field Management
- Daily Logs API available via Data Connector
- Meeting Minutes extraction supported
- Custom field reports

### Assets API
- Track equipment, materials
- Link to locations, issues

---

## BIM 360 and ACC APIs

### Platform Evolution
- **BIM 360**: Original cloud construction management platform
- **ACC**: Unified platform including Autodesk Docs, Autodesk Build, Model Coordination

### BIM 360 API Modules
1. **Document Management** - Files, folders, versions
2. **Model Coordination** - Clash detection, coordination
3. **Field** - Mobile field data capture
4. **Project Management** - RFIs, submittals
5. **Account Admin** - Users, companies, projects

### ACC-Specific APIs
1. **Issues** - Construction issues tracking
2. **RFIs** - Requests for information
3. **Submittals** - Shop drawings, specs review
4. **Photos** - Photo documentation
5. **Assets** - Equipment/material tracking
6. **Locations** - Location hierarchy
7. **Checklists** - Quality/safety checklists
8. **Cost Management** - Budget, contracts
9. **Daily Logs** - Daily reports

### API Compatibility Notes
- `/bim360/` endpoints work for both BIM 360 and ACC
- `/construction/` endpoints are ACC-specific
- Switch from `/construction/` to `/bim360/` for RFIs

### Viewer API
- Visualize 60+ file formats in browser
- 2D sheets and 3D models
- No plugins required
- Markup and annotation support

---

## Webhook Capabilities

### Overview
- **Documentation**: https://aps.autodesk.com/en/docs/webhooks/v1
- Real-time event notifications
- Callback URL receives POST with event payload

### Creating Webhooks
```http
POST https://developer.api.autodesk.com/webhooks/v1/systems/{system}/events/{event}/hooks
Content-Type: application/json
x-ads-region: US

{
  "callbackUrl": "https://your-app.com/webhook",
  "scope": {
    "folder": "urn:adsk.wipstg:fs.folder:..."
  }
}
```

### Supported Systems and Events

#### Data Management (dm)
- `dm.version.added` - New version uploaded
- `dm.version.modified` - Version metadata changed
- `dm.version.deleted` - Version deleted
- `dm.folder.added` - Folder created
- `dm.folder.modified` - Folder updated

#### ACC Issues
- **System**: `autodesk.construction.issues`
- `issue.created` - Issue created
- `issue.updated` - Issue modified (includes changedAttributes)
- `issue.deleted` - Issue removed
- **Regions**: US, EMEA, AUS (all regions as of Sept 2025)

#### ACC Reviews
- Review creation events
- Review closure events
- Only participants and admins receive notifications

### Webhook Security
- Secret token support for payload signing
- Hash signature in `x-adsk-signature` header
- Verify signatures to prevent callback spoofing

### Delivery Guarantees
- At-least-once delivery when no errors
- 4 retries over 48+ hours on failure
- Webhook disabled after final retry failure

### Regional Headers
- **Required**: `region` or `x-ads-region` header
- Values: US (default), EMEA, AUS

---

## Rate Limits and Access Tiers

### Rate Limit Behavior
- **Response Code**: 429 Too Many Requests
- **Header**: `Retry-After` indicates wait time
- Implement exponential backoff for retries
- Cache infrequently changing data

### Best Practices
1. Monitor rate limit headers
2. Implement exponential backoff
3. Cache static data (folder contents, project metadata)
4. Use batch endpoints where available
5. Request rate limit increases if needed

### Requesting Rate Limit Increases
1. Download "Rate Limit change request form" from ADN
2. Analyze current usage patterns
3. Provide reasonable justification
4. Submit via ADN member support ticket

### Access Tiers (December Model)

#### Free Tier
- Monthly access to cloud APIs
- Ideal for testing, POCs, learning
- Usage caps on rated APIs:
  - Automation API
  - Model Derivative API
  - Flow Graph Engine API
  - Reality Capture API
- Suspended when caps exceeded (resumes next month)

#### Paid Tier
- Token-based billing
- Minimum purchase: 100 tokens
- Tokens expire after 1 year
- Metered usage for rated APIs
- Same API access as Free tier

### API-Specific Limits

| API Category | Notes |
|--------------|-------|
| Data Management | Standard rate limits |
| ACC Issues | 3-legged token only |
| Model Derivative | Rated API (token consumption) |
| Viewer | Included with data access |
| Webhooks | Standard limits |
| Data Connector | Max 50 projects per request |

---

## API Reference Links

### Official Documentation
- **APS Portal**: https://aps.autodesk.com
- **ACC APIs Overview**: https://aps.autodesk.com/en/docs/acc/v1/overview/
- **ACC API Reference**: https://aps.autodesk.com/en/docs/acc/v1/reference
- **BIM 360 API**: https://aps.autodesk.com/en/docs/bim360/v1
- **OAuth Documentation**: https://aps.autodesk.com/en/docs/oauth/v2
- **Webhooks API**: https://aps.autodesk.com/en/docs/webhooks/v1
- **Data Management API**: https://aps.autodesk.com/data-management-api

### Legacy PlanGrid
- **Developer Portal**: https://developer.plangrid.com
- **Support**: plangrid-api@autodesk.com
- **Help Center**: https://help.plangrid.com

### SDKs and Tools
- **Node.js SDK**: https://github.com/autodesk-platform-services/aps-sdk-node
- **Python SDK (Community)**: https://github.com/realdanielbyrne/acc_sdk
- **Postman Collections**:
  - Issues: https://github.com/autodesk-platform-services/aps-acc.issues.api-postman.collection
  - Data Connector: https://github.com/autodesk-platform-services/aps-data-connector-postman.collection

### Sample Applications
- **Webhook Notifier**: https://github.com/autodesk-platform-services/aps-webhook-notifier
- **Data Connector Dashboard**: https://github.com/autodesk-platform-services/aps-data-connector-dashboard
- **Hubs Browser**: https://tutorials.autodesk.io/tutorials/hubs-browser/

### Getting Started Tutorials
- https://aps.autodesk.com/en/docs/acc/v1/tutorials/getting-started
- https://tutorials.autodesk.io

---

## 2025-2026 Updates and New Features

### Recent ACC Updates
1. **OCR for PDF Title Blocks** - Extract attributes automatically
2. **AutoCAD Web Integration** - Edit DWG files in browser
3. **Autodesk Assistant (Beta)** - AI-powered search and queries
4. **Takeoff Specification Tool** - Aligned with Docs and Build data
5. **Enhanced Submittals API** - Full CRUD operations, transitions

### API Evolution
- Migration from Forge branding to APS
- Consolidated `/bim360/` endpoints for ACC compatibility
- New two-tier pricing model
- Expanded webhook event coverage
- Data Connector multi-project support

---

## Recommendations for Integration Development

### For New Projects
1. **Use APS/ACC APIs** - Not legacy PlanGrid
2. **Implement 3-legged OAuth with PKCE** - Best security practice
3. **Plan for rate limits** - Implement caching and backoff
4. **Use webhooks** - Real-time updates vs polling
5. **Leverage Data Connector** - For bulk data extraction

### Authentication Strategy
```
1. Register app at aps.autodesk.com
2. Select "BIM 360" API type
3. Get client ID and secret
4. Add custom integration in ACC admin
5. Implement both 2-legged and 3-legged flows
6. Store tokens securely, auto-refresh
```

### Data Sync Patterns
- **Real-time**: Webhooks for immediate notifications
- **Batch**: Data Connector API for scheduled extracts
- **On-demand**: REST APIs for specific queries

---

*Research compiled from Autodesk official documentation and developer resources.*
