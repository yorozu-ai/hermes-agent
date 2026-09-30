"""Tests for MCP dynamic tool discovery (notifications/tools/list_changed)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_common import mcp_tool_registration_fingerprint
from tools.mcp_tool_registration import _register_server_tools
from tools.registry import ToolRegistry


def _make_mcp_tool(name: str, desc: str = ""):
    return SimpleNamespace(name=name, description=desc, inputSchema=None)


class TestRegisterServerTools:
    """Tests for the extracted _register_server_tools helper."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    def test_exposes_live_server_aliases(self, mock_registry):
        """Registered MCP tools are reachable via live raw-server aliases."""
        server = MCPServerTask("my_srv")
        server._tools = [_make_mcp_tool("my_tool", "desc")]
        server.session = MagicMock()
        from toolsets import resolve_toolset, validate_toolset

        with patch("tools.registry.registry", mock_registry):
            registered = _register_server_tools("my_srv", server, {})
            assert "mcp__my_srv__my_tool" in registered
            assert "mcp__my_srv__my_tool" in mock_registry.get_all_tool_names()
            assert validate_toolset("my_srv") is True
            assert "mcp__my_srv__my_tool" in resolve_toolset("my_srv")

    def test_colliding_static_toolset_name_merges_both_tool_sets(self, mock_registry):
        """An MCP server named after a built-in toolset must not be shadowed.

        Regression: an MCP server registered as `homeassistant` (colliding
        with the static `homeassistant` toolset) had its tools silently
        dropped because get_toolset() returned the static definition without
        consulting the alias registered by _register_server_tools().
        """
        from toolsets import TOOLSETS, get_toolset, resolve_toolset

        assert "homeassistant" in TOOLSETS  # collision premise
        static_tools = set(TOOLSETS["homeassistant"]["tools"])

        server = MCPServerTask("homeassistant")
        server._tools = [_make_mcp_tool("get_entities", "List HA entities")]
        server.session = MagicMock()

        with patch("tools.registry.registry", mock_registry):
            registered = _register_server_tools("homeassistant", server, {})
            assert "mcp__homeassistant__get_entities" in registered

            ts = get_toolset("homeassistant")
            # Static built-ins are still present...
            assert static_tools <= set(ts["tools"])
            # ...and the MCP server's tools are no longer shadowed.
            assert "mcp__homeassistant__get_entities" in ts["tools"]
            assert "mcp__homeassistant__get_entities" in resolve_toolset("homeassistant")


class TestRefreshTools:
    """Tests for MCPServerTask._refresh_tools nuke-and-repave cycle."""

    @pytest.fixture
    def mock_registry(self):
        return ToolRegistry()

    @pytest.mark.asyncio
    async def test_nuke_and_repave(self, mock_registry):
        """Old tools are removed and new tools registered on refresh."""
        server = MCPServerTask("live_srv")
        server._refresh_lock = asyncio.Lock()
        server._config = {}
        from toolsets import resolve_toolset

        # Seed initial state: one old tool registered
        mock_registry.register(
            name="mcp__live_srv__old_tool", toolset="mcp-live_srv", schema={},
            handler=lambda x: x, check_fn=lambda: True, is_async=False,
            description="", emoji="",
        )
        server._registered_tool_names = ["mcp__live_srv__old_tool"]

        # New tool list from server
        new_tool = _make_mcp_tool("new_tool", "new behavior")
        server.session = SimpleNamespace(
            list_tools=AsyncMock(
                return_value=SimpleNamespace(tools=[new_tool])
            )
        )

        with patch("tools.registry.registry", mock_registry):
            await server._refresh_tools()
            assert "mcp__live_srv__old_tool" not in mock_registry.get_all_tool_names()
            assert "mcp__live_srv__old_tool" not in resolve_toolset("live_srv")
            assert "mcp__live_srv__new_tool" in mock_registry.get_all_tool_names()
            assert "mcp__live_srv__new_tool" in resolve_toolset("live_srv")
            assert server._registered_tool_names == ["mcp__live_srv__new_tool"]

    @pytest.mark.asyncio
    async def test_refreshes_registration_schema_under_a_stable_name(self):
        """Input changes refresh registration; output-only changes do not churn it."""
        server = MCPServerTask("live_srv")
        server._config = {}
        old_tool = _make_mcp_tool("contact_create")
        old_tool.inputSchema = {"type": "object", "properties": {"name": {"type": "string"}}}
        old_tool.outputSchema = {"type": "object", "properties": {"contact": {}}}
        old_tool.annotations = {"readOnlyHint": True, "title": "old"}
        server._tools = [old_tool]
        server._tool_registration_fingerprint = mcp_tool_registration_fingerprint(server._tools)
        server._registered_tool_names = ["mcp__live_srv__contact_create"]
        new_tool = _make_mcp_tool("contact_create")
        new_tool.inputSchema = old_tool.inputSchema
        new_tool.outputSchema = {
            "type": "object",
            "properties": {"contact": {"properties": {"race": {"type": "string"}}}},
        }
        new_tool.annotations = {"readOnlyHint": True, "title": "new"}
        with patch(
            "tools.mcp_tool_registration._register_server_tools",
            return_value=server._registered_tool_names,
        ) as register:
            server._tool_manifest_revision = 2
            server._tool_manifest_applied_revision = 2
            stale_tool = _make_mcp_tool("contact_create")
            stale_tool.inputSchema = {"type": "object"}
            assert await server._refresh_tools(
                new_mcp_tools=[stale_tool], manifest_revision=1
            ) is False
            register.assert_not_called()
            assert server._tools == [old_tool]

            assert await server._refresh_tools(new_mcp_tools=[old_tool]) is False
            register.assert_not_called()
            assert await server._refresh_tools(new_mcp_tools=[new_tool]) is False
            register.assert_not_called()
            new_tool.inputSchema = {
                "type": "object",
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            }
            assert await server._refresh_tools(new_mcp_tools=[new_tool]) is True
            register.assert_called_once()

        assert server._tools == [new_tool]
        assert server._tool_registration_fingerprint == mcp_tool_registration_fingerprint([new_tool])

    @pytest.mark.asyncio
    async def test_discards_manifest_that_finishes_after_session_reconnect(self):
        """A delayed old-session response cannot replace a newly discovered manifest."""
        server = MCPServerTask("live_srv")
        server._config = {}
        server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(tools=SimpleNamespace()))
        old_session = SimpleNamespace()
        new_session = SimpleNamespace()
        server.session = old_session
        server._session_epoch = 1

        async def list_tools():
            server.session = new_session
            server._session_epoch = 2
            return SimpleNamespace(tools=[_make_mcp_tool("stale_tool")])

        old_session.list_tools = list_tools
        await server._refresh_tools()

        assert server._tools == []
        assert server._tool_manifest_revision == 0


    @pytest.mark.asyncio
    async def test_suspect_health_refreshes_registration_from_manifest(self):
        """A successful suspect-session probe applies changed input schemas."""
        server = MCPServerTask("live_srv")
        server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(tools=SimpleNamespace()))
        server.session = SimpleNamespace()
        server._suspect_reason = "keepalive failed"
        manifest = [_make_mcp_tool("contact_create")]

        with patch.object(MCPServerTask, "_keepalive_probe", new=AsyncMock(return_value=manifest)) as probe:
            with patch.object(MCPServerTask, "_refresh_tools", new=AsyncMock()) as refresh:
                assert await server.ensure_healthy() is True

        probe.assert_awaited_once()
        refresh.assert_awaited_once_with(
            new_mcp_tools=manifest,
            manifest_revision=server._tool_manifest_revision,
            manifest_epoch=server._session_epoch,
        )
        assert server._suspect_reason is None


@pytest.mark.asyncio
async def test_mcp_sdk_accepts_new_output_schema_after_tools_list_refresh():
    """A fresh tools/list updates the SDK validator used by the next tools/call."""
    from mcp import ClientSession
    from mcp.types import CallToolResult, ListToolsResult, Tool

    def output_schema(*fields):
        properties = {"id": {"type": "string"}}
        properties.update({
            field: {"type": "number" if field == "total_premium" else "string"}
            for field in fields
        })
        return {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "object",
                    "properties": properties,
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
            "required": ["contact"],
            "additionalProperties": False,
        }

    old_tool = Tool(name="contact_create", input_schema={}, output_schema=output_schema())
    new_tool = Tool(
        name="contact_create",
        input_schema={},
        output_schema=output_schema("race", "client_tier", "total_premium"),
    )
    result = CallToolResult(
        content=[],
        structured_content={
            "contact": {
                "id": "1",
                "race": "x",
                "client_tier": "gold",
                "total_premium": 100.0,
            },
        },
        is_error=False,
    )
    session = ClientSession(dispatcher=object())
    session.send_request = AsyncMock(side_effect=[
        ListToolsResult(tools=[old_tool]),
        result,
        ListToolsResult(tools=[new_tool]),
        result,
    ])

    await session.list_tools()
    with pytest.raises(RuntimeError, match="Invalid structured content") as failure:
        await session.call_tool("contact_create", arguments={})
    assert "client_tier" in str(failure.value)

    await session.list_tools()
    refreshed = await session.call_tool("contact_create", arguments={})
    assert refreshed.structured_content == result.structured_content


class TestMessageHandler:
    """Tests for MCPServerTask._make_message_handler dispatch."""

    @pytest.mark.asyncio
    async def test_dispatches_tool_list_changed(self):
        from tools.mcp_tool import _MCP_NOTIFICATION_TYPES
        if not _MCP_NOTIFICATION_TYPES:
            pytest.skip("MCP SDK ToolListChangedNotification not available")

        from mcp.types import ServerNotification, ToolListChangedNotification

        server = MCPServerTask("notif_srv")
        # Product now schedules the refresh as a background task (see
        # _schedule_tools_refresh in mcp_tool.py ~L918) rather than awaiting
        # it directly, to avoid wedging the stdio JSON-RPC stream. Patch at
        # the scheduler seam so we can still assert dispatch happened without
        # reaching into asyncio.create_task internals.
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
            handler = server._make_message_handler()
            notification = ToolListChangedNotification(
                method="notifications/tools/list_changed"
            )
            if hasattr(ServerNotification, "model_validate"):
                # mcp < 2.0 wrapped notifications in a RootModel; 2.0 made
                # ServerNotification a plain union of the concrete types, which
                # has no constructor to wrap with.
                notification = ServerNotification(root=notification)
            await handler(notification)
            mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_exceptions_and_other_messages(self):
        server = MCPServerTask("notif_srv")
        with patch.object(MCPServerTask, "_schedule_tools_refresh") as mock_schedule:
            handler = server._make_message_handler()
            # Exceptions should not trigger refresh
            await handler(RuntimeError("connection dead"))
            # Unknown message types should not trigger refresh
            await handler({"jsonrpc": "2.0", "result": "ok"})
            mock_schedule.assert_not_called()


class TestDeregister:
    """Tests for ToolRegistry.deregister."""

    def test_removes_tool(self):
        reg = ToolRegistry()
        reg.register(name="foo", toolset="ts1", schema={}, handler=lambda x: x)
        assert "foo" in reg.get_all_tool_names()
        reg.deregister("foo")
        assert "foo" not in reg.get_all_tool_names()


    def test_noop_for_unknown_tool(self):
        reg = ToolRegistry()
        reg.deregister("nonexistent")  # Should not raise
