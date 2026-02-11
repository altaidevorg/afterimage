import json
import subprocess
import os
import requests
import time
import asyncio
from typing import List, Dict, Any, Optional
from fastmcp.client import Client, SSETransport, StreamableHttpTransport
from fastmcp.mcp_config import infer_transport_type_from_url

# Ensure ~/.local/bin is in PATH for tools like uv
local_bin = os.path.expanduser("~/.local/bin")
if local_bin not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ.get('PATH', '')}"

class MCPClient:
    """
    A simple MCP client that connects to an MCP server via stdio,
    initializes the connection, fetches tools, and disconnects.
    """
    
    def __init__(self, command: str, args: List[str] = None, env: Dict[str, str] = None):
        self.command = command
        self.args = args or []
        self.env = env or os.environ.copy()
        self.process = None
        self._request_id = 0

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method
        }
        if params:
            request["params"] = params
        
        self._request_id += 1
        
        # Write to stdin
        json_req = json.dumps(request) + "\n"
        self.process.stdin.write(json_req.encode('utf-8'))
        self.process.stdin.flush()
        
        # Read from stdout until we get a matching response
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed connection unexpectedly")
            
            try:
                response = json.loads(line.decode('utf-8'))
            except json.JSONDecodeError:
                continue
                
            if response.get("id") == request["id"]:
                if "error" in response:
                    raise RuntimeError(f"MCP Error: {response['error']}")
                return response.get("result", {})

    def fetch_tools(self) -> List[Dict[str, Any]]:
        """
        Connects to the MCP server, initializes, lists tools, and disconnects.
        Returns a list of tool definitions in MCP format.
        """
        try:
            # Start the process
            cmd = [self.command] + self.args
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env
            )
            
            # 1. Initialize
            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "afterimage-client",
                    "version": "1.0.0"
                }
            }
            self._send_request("initialize", init_params)
            
            # 2. Send initialized notification
            notify = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            self.process.stdin.write((json.dumps(notify) + "\n").encode('utf-8'))
            self.process.stdin.flush()
            
            # 3. List Tools
            result = self._send_request("tools/list")
            tools = result.get("tools", [])
            
            return tools
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tools from MCP server: {str(e)}")
            
        finally:
            # Cleanup
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()


class RemoteMCPClient:
    """
    A simple MCP client that connects to a Remote MCP server via SSE/HTTP using fastmcp.
    """
    
    def __init__(self, url: str, headers: Dict[str, str] = None):
        self.url = url
        self.headers = headers or {}

    async def _fetch_tools_async(self) -> List[Dict[str, Any]]:
        # Infer transport type and construct transport with headers
        try:
            transport_type = infer_transport_type_from_url(self.url)
            if transport_type == "sse":
                transport = SSETransport(self.url, headers=self.headers)
            else:
                transport = StreamableHttpTransport(self.url, headers=self.headers)
        except Exception:
            # Fallback to SSE if inference fails but URL is provided
            transport = SSETransport(self.url, headers=self.headers)

        # Connect to the MCP server using fastmcp Client with custom transport
        client = Client(transport=transport)
        async with client:
            tools = await client.list_tools()
            # Convert tools to dictionary format expected by the application
            return [tool.model_dump() for tool in tools]

    def fetch_tools(self) -> List[Dict[str, Any]]:
        """Connect, initialize, and fetch tools."""
        try:
            return asyncio.run(self._fetch_tools_async())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tools from Remote MCP: {str(e)}")


class MCPConfigClient:
    """
    An MCP client that connects using an MCP configuration dictionary.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def _fetch_tools_async(self) -> List[Dict[str, Any]]:
        # Connect using the config
        client = Client(self.config)
        async with client:
            tools = await client.list_tools()
            return [tool.model_dump() for tool in tools]

    def fetch_tools(self) -> List[Dict[str, Any]]:
        """Connect and fetch tools."""
        try:
            return asyncio.run(self._fetch_tools_async())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tools from MCP Config: {str(e)}")


def mcp_tool_to_function_def(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an MCP tool definition to the format expected by Afterimage/OpenAI.
    """
    return {
        "name": mcp_tool.get("name"),
        "description": mcp_tool.get("description", ""),
        "parameters": mcp_tool.get("inputSchema", {}),
        "required": mcp_tool.get("inputSchema", {}).get("required", [])
    }
