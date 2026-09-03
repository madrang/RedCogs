"""The MCP resource tools: the bridge between the agent and the resources of the MCP servers."""


class MCPTools:
    """The list_resources and read_resource tools.

    MCP resources are application-driven: the protocol gives the model no
    way to call resources/list or resources/read itself, so these tools are
    the bridge. The agent picks a resource, the harness sends the client
    request.
    """

    def mcp_tools(self) -> list:
        """The OpenAI function schemas of the MCP resource tools."""
        if self.mcp is None:
            return []
        # Offered without the `mcp` package too: the built-in harness set
        # needs no client. The remote servers answer an unavailable error.
        return [
            {
                "type": "function"
                , "function": {
                    "name": "list_resources"
                    , "description": (
                        "List the resources of the connected MCP servers and of the built-in harness server. "
                        "A resource is data published as context, such as a file or a schema. "
                        "Each entry has a URI, a name, a MIME type, and a description. "
                        "Read one entry with read_resource."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "server": {
                                "type": "string"
                                , "description": "The name of one MCP server. Blank lists every server."
                                , "default": ""
                            }
                        }
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "read_resource"
                    , "description": (
                        "Read one resource. The server name and the URI come from list_resources. "
                        "A text resource returns its text. A binary resource reports its type and its size."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "server": {
                                "type": "string"
                                , "description": "The name of the server that serves the resource. The built-in reference files use \"harness\"."
                            }
                            , "uri": {
                                "type": "string"
                                , "description": "The URI of the resource, from list_resources."
                            }
                        }
                        , "required": ["server", "uri"]
                    }
                }
            }
        ]

    async def _tool_list_resources(self, arguments: dict, **_scope) -> str:
        if self.mcp is None:
            return "Error: the MCP manager is not wired."
        server = arguments.get("server")
        if server is not None and not isinstance(server, str):
            return "Error: the server must be a string."
        return await self.mcp.list_resources((server or "").strip())

    async def _tool_read_resource(self, arguments: dict, **scope) -> str:
        if self.mcp is None:
            return "Error: the MCP manager is not wired."
        server = arguments.get("server")
        uri = arguments.get("uri")
        if not isinstance(server, str) or not server.strip():
            return "Error: the server must be a non-empty string."
        if not isinstance(uri, str) or not uri.strip():
            return "Error: the uri must be a non-empty string."
        # The live status resource reports the context of the reading
        # session: the channel of a guild message, the user of a direct
        # message, the same rule as the reply engine.
        session_id = scope.get("channel_id") if scope.get("guild_id") is not None else scope.get("user_id")
        return await self.mcp.read_resource(server.strip(), uri.strip(), session_id=session_id)
