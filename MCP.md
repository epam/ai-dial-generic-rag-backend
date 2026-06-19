# Generic RAG MCP Setup

MCP server for coding agents such as Cursor or Claude Code.

> **Note:** The available tools are subject to change.

## Prerequisites

Install dependencies:

   ```bash
   make install
   ```

## Start MCP Server

Start the application (with DIAL core via docker-compose) to run the MCP server:

```bash
make main
```

## MCP Endpoint URL

The MCP endpoint is accessed using following URL:

```
{DIAL_URL}/v1/deployments/{application_id}/mcp
```

**NOTE**: older DIAL core versions expect `toolset` instead of `deployments` in the URL:

```
{DIAL_URL}/v1/toolset/{application_id}/mcp
```

### How to determine the `application_id`

The `application_id` depends on how the application was created:

- **Organization apps** (defined in DIAL core config): the deployment name itself, e.g. `generic-rag-example`

  ```
  {DIAL_URL}/v1/deployments/generic-rag-example/mcp
  ```

- **User bucket apps** (created via [API](https://dialx.ai/dial_api#tag/Applications/operation/saveCustomApplication)): `applications/{bucket}/{app_name}`

  ```
  {DIAL_URL}/v1/deployments/applications/{bucket}/{app_name}/mcp
  ```

  For example, if bucket is `123` and app name is `generic-rag-app`:

  ```
  {DIAL_URL}/v1/deployments/applications/123/generic-rag-app/mcp
  ```

> **Tip:** To determine your bucket, run:
>
> ```bash
> curl -H "api-key: <DIAL_API_KEY>" {DIAL_URL}/v1/bucket
> ```
>
> Use the `bucket` value from the response to construct the URL.

## Coding Agent Configuration

### Cursor

1. Go to **Settings** → **Tools and MCP** → **Add New MCP Server**
2. Add the following configuration (replace URL with your DIAL deployment URL):

   ```json
   {
     "mcpServers": {
       "generic-rag-mcp": {
         "type": "http",
         "url": "{DIAL_URL}/v1/deployments/{application_id}/mcp",
         "headers": {
           "api-key": "<DIAL_API_KEY>"
         }
       }
     }
   }
   ```

3. Verify the MCP was successfully loaded and its status is green.
   If not, ensure the app is running and try to reload the MCP.

### Claude Code

1. Add the MCP server via terminal (replace URL with your DIAL deployment URL):

   ```bash
   claude mcp add \
      --transport http \
      generic-rag-mcp \
      {DIAL_URL}/v1/deployments/{application_id}/mcp \
      --header "api-key: <DIAL_API_KEY>"
   ```

2. Verify it's connected:

   ```bash
   claude mcp list
   claude mcp get generic-rag-mcp
   ```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_channel_config` | Retrieve the full channel configuration for the current DIAL application |
