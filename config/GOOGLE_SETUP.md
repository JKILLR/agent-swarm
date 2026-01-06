# Google Workspace Integration Setup

This guide walks you through setting up Google API access for Gmail, Calendar, Drive, and Tasks.

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Click "Select a project" dropdown at the top
3. Click "New Project"
4. Name it "Agent Swarm Integration" (or any name you prefer)
5. Click "Create"

## Step 2: Enable APIs

Go to **APIs & Services > Library** and enable each of these:
- Gmail API
- Google Calendar API
- Google Drive API
- Google Tasks API

Search for each one and click "Enable".

## Step 3: Configure OAuth Consent Screen

1. Go to **APIs & Services > OAuth consent screen**
2. Select **External** user type (unless you have Google Workspace)
3. Fill in the form:
   - App name: "Agent Swarm"
   - User support email: Your email
   - Developer contact: Your email
4. Click "Save and Continue"
5. On Scopes page, click "Add or Remove Scopes"
6. Add these scopes:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.send`
   - `https://www.googleapis.com/auth/gmail.modify`
   - `https://www.googleapis.com/auth/calendar.readonly`
   - `https://www.googleapis.com/auth/calendar.events`
   - `https://www.googleapis.com/auth/drive.readonly`
   - `https://www.googleapis.com/auth/drive.file`
   - `https://www.googleapis.com/auth/tasks.readonly`
   - `https://www.googleapis.com/auth/tasks`
7. Click "Save and Continue"
8. Add yourself as a test user (your Google email)
9. Click "Save and Continue"

**IMPORTANT**: To avoid 7-day token expiration:
- Go back to OAuth consent screen
- Click "Publish App"
- Confirm to move to production

## Step 4: Create OAuth Credentials

1. Go to **APIs & Services > Credentials**
2. Click "Create Credentials" > "OAuth client ID"
3. Application type: **Desktop app** (or Web application)
4. Name: "Agent Swarm Desktop"
5. If Web application:
   - Add redirect URI: `http://localhost:8000/api/google/callback`
6. Click "Create"
7. **Download JSON** - save as `config/google_credentials.json`

## Step 5: Configure the Integration

### Option A: Using credentials file (recommended)

1. Place the downloaded JSON file at:
   ```
   config/google_credentials.json
   ```

### Option B: Using environment variables

Add to `backend/.env`:
```
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
```

## Step 6: Authenticate

1. Start the backend server:
   ```bash
   cd backend && python main.py
   ```

2. Check status:
   ```bash
   curl http://localhost:8000/api/google/status
   ```

3. Get the auth URL:
   ```bash
   curl http://localhost:8000/api/google/auth/url
   ```

4. Open the `auth_url` in your browser

5. Sign in with your Google account and grant permissions

6. You'll be redirected back. The token is saved automatically.

7. Verify authentication:
   ```bash
   curl http://localhost:8000/api/google/status
   ```

## API Endpoints Reference

### Authentication
- `GET /api/google/status` - Check auth status
- `GET /api/google/auth/url` - Get OAuth URL
- `GET /api/google/callback?code=XXX` - OAuth callback (automatic)

### Gmail
- `GET /api/google/gmail/messages` - List emails
  - `?max_results=10` - Limit results
  - `?query=is:unread` - Gmail search query
  - `?label_ids=INBOX,UNREAD` - Filter by labels
- `GET /api/google/gmail/messages/{id}` - Get full email
- `POST /api/google/gmail/send` - Send email
- `POST /api/google/gmail/messages/{id}/read` - Mark as read

### Calendar
- `GET /api/google/calendar/list` - List calendars
- `GET /api/google/calendar/events` - Get events
  - `?calendar_id=primary`
  - `?time_min=2025-01-01T00:00:00`
  - `?time_max=2025-01-08T00:00:00`
  - `?max_results=10`
- `POST /api/google/calendar/events` - Create event

### Drive
- `GET /api/google/drive/files` - List files
  - `?query=name contains 'report'`
  - `?folder_id=XXX`
  - `?max_results=10`
- `GET /api/google/drive/files/{id}/content` - Get file content

### Tasks
- `GET /api/google/tasks/lists` - Get task lists
- `GET /api/google/tasks` - Get tasks
  - `?task_list_id=@default`
  - `?show_completed=false`
- `POST /api/google/tasks` - Create task
- `POST /api/google/tasks/{id}/complete` - Complete task

## Security Notes

- Tokens are stored in `config/google_token.json`
- Add this file to `.gitignore`:
  ```
  config/google_token.json
  config/google_credentials.json
  ```
- Refresh tokens are used automatically
- Publishing to production avoids 7-day token expiration

## Troubleshooting

### "Token expired" errors
- Run the auth flow again
- Make sure app is published to production

### "Insufficient permissions"
- Check that all APIs are enabled
- Verify scopes were added correctly
- Re-authorize to get new token with updated scopes

### "redirect_uri_mismatch"
- Make sure redirect URI in Google Console matches exactly:
  `http://localhost:8000/api/google/callback`
