# Spotify Authorization Setup

## Steps
1. Login to Spotify Developer Dashboard.
2. Create a new app → copy Client ID and Client Secret.
3. Add Redirect URI: https://shawanna-unbattered-lexie.ngrok-free.dev/callback
4. Run the login flow to get an authorization code.
5. Exchange code for tokens (Access + Refresh).
6. Store in `.env` file:
   SPOTIFY_CLIENT_ID=xxxx
   SPOTIFY_CLIENT_SECRET=xxxx
   SPOTIFY_REFRESH_TOKEN=xxxx

## Refreshing Tokens
- Access tokens expire after 1 hour.
- `spotify_refresh_token.py` automatically fetches a new token using the refresh token.

## Scopes Used
- playlist-read-private
- streaming
- user-modify-playback-state
- user-read-email
- user-read-private

## Secret Rotation
- If Client Secret changes → update `.env`.
- If Refresh Token becomes invalid → re-run login flow to generate new one.