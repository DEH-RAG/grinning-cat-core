"""Integration tests for the ``mgmt_message`` core plugin.

Coverage:
(a) storage round-trip in the global ``system:plugins:mgmt_message`` key
    (the default framework mechanism for the system agent — no overrides),
(b) the management gate: the real ``auth_request`` hook denies unprivileged
    principals when ``management_active`` is true, and the denial is translated
    by ``ConnectionAuth`` into ``CustomForbiddenException`` (HTTP) /
    ``WebSocketException(code=1008)`` (WS),
(c) ``GET /plugins/settings/mgmt_message`` returns the 4 settings in normal
    mode (the RITA read path),
(d) ``management_active=false`` is a no-op for any principal.

Uses the ``tests/conftest.py`` fixtures: Redis db=1 (isolated), agent
``"agent_test"``, mocked Qdrant, synchronous background tasks. No live
LLM/embedder required.
"""

from types import SimpleNamespace

import pytest
from fastapi import WebSocketException

from cat.auth.connection import AuthorizedInfo, HTTPAuth, WebSocketAuth
from cat.auth.permissions import (
    AuthPermission,
    AuthResource,
    AuthUserInfo,
    get_base_permissions,
)
from cat.core_plugins.mgmt_message.plugin import auth_request
from cat.db.cruds import plugins as crud_plugins
from cat.db.database import DEFAULT_PLUGINS_KEY, DEFAULT_SYSTEM_KEY, get_sync_db
from cat.exceptions import CustomForbiddenException

# the plugin id (folder name) and the global key where its settings live
PLUGIN_ID = "mgmt_message"
MGMT_GLOBAL_KEY = f"{DEFAULT_SYSTEM_KEY}:{DEFAULT_PLUGINS_KEY}:{PLUGIN_ID}"


def _make_user():
    """A normal chat user: no SYSTEM permission."""
    return AuthUserInfo(id="user", name="User", permissions=get_base_permissions())


def _make_admin_user():
    return AuthUserInfo(
        id="admin",
        name="Admin",
        permissions={str(AuthResource.SYSTEM): [str(AuthPermission.WRITE)]},
    )


async def _store(payload: dict):
    await crud_plugins.set_setting(DEFAULT_SYSTEM_KEY, PLUGIN_ID, payload)


async def _cleanup():
    await crud_plugins.delete_setting(DEFAULT_SYSTEM_KEY, PLUGIN_ID)


# ---------------------------------------------------------------------------
# (a) storage round-trip in the global system:plugins:mgmt_message key
# ---------------------------------------------------------------------------

async def test_storage_round_trip_in_system_plugins():
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": True,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }

    await _store(payload)

    loaded = await crud_plugins.get_setting(DEFAULT_SYSTEM_KEY, PLUGIN_ID)
    assert loaded == payload

    # the value is stored under the global system:plugins:<id> key, not an agents:* key
    db = get_sync_db()
    assert db.exists(MGMT_GLOBAL_KEY) == 1

    agent_keys = db.keys("agents:*")
    assert agent_keys == []

    await _cleanup()


# ---------------------------------------------------------------------------
# (b) management gate: the real auth_request hook
# ---------------------------------------------------------------------------

async def test_auth_request_denies_unprivileged_when_active():
    message = "Sistema in manutenzione"

    # store via the default framework mechanism, then let the real hook read
    # the same value (no monkeypatching)
    await _store({"management_message": message, "management_active": True})

    result = await auth_request.function(_make_user(), DEFAULT_SYSTEM_KEY, None)

    assert result == message

    await _cleanup()


async def test_auth_request_allows_system_principal_when_active():
    await _store({"management_message": "Sistema in manutenzione", "management_active": True})

    result = await auth_request.function(_make_admin_user(), DEFAULT_SYSTEM_KEY, None)

    assert result is None

    await _cleanup()


# ---------------------------------------------------------------------------
# (b) management gate E2E: real hook wired through ConnectionAuth
# ---------------------------------------------------------------------------

class _RealHookPluginManager:
    """Plugin-manager stand-in that executes the real ``auth_request`` hook."""

    def __init__(self, hooks):
        self.hooks = hooks

    async def execute_hook(self, hook_name, *args, **kwargs):
        tea_cup = args[0]
        for hook in self.hooks[hook_name]:
            result = await hook.function(tea_cup, *args[1:], **kwargs)
            if result is not None:
                tea_cup = result
        return tea_cup


class _FakeCoreAuthHandler:
    def __init__(self, user):
        self._user = user

    async def authorize(self, connection, resource, permission, agent_key):
        return self._user


class _FakeLizard:
    """Minimal stand-in for BillTheLizard (same shape as test_mgmt_hook_gateway)."""

    def __init__(self, plugin_manager, user):
        self.plugin_manager = plugin_manager
        self.agent_key = DEFAULT_SYSTEM_KEY
        self.core_auth_handler = _FakeCoreAuthHandler(user)

    async def get_cheshire_cat(self, agent_id):
        return None

    def is_custom_endpoint(self, url_path):
        return False


class _FakeConnection:
    def __init__(self, scope_type="http"):
        self.scope = {"type": scope_type}
        self.url = SimpleNamespace(path="/test")
        self.path_params = {}
        self.query_params = {}
        self.headers = {}
        self.app = SimpleNamespace(state=SimpleNamespace(lizard=None))


def _make_connection(lizard, scope_type="http"):
    connection = _FakeConnection(scope_type=scope_type)
    connection.app.state.lizard = lizard
    return connection


def _make_lizard_with_real_hook(user):
    return _FakeLizard(_RealHookPluginManager({"auth_request": [auth_request]}), user)


async def test_http_gateway_denial_with_real_hook(monkeypatch):
    message = "Sistema in manutenzione"

    async def fake_get_setting(agent_id, plugin_id):
        return {"management_active": True, "management_message": message}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    lizard = _make_lizard_with_real_hook(_make_user())
    connection = _make_connection(lizard, scope_type="http")

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(CustomForbiddenException) as exc_info:
        await auth(connection)

    assert exc_info.value.args[0] == message


async def test_websocket_gateway_denial_with_real_hook(monkeypatch):
    message = "Sistema in manutenzione"

    async def fake_get_setting(agent_id, plugin_id):
        return {"management_active": True, "management_message": message}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    lizard = _make_lizard_with_real_hook(_make_user())
    connection = _make_connection(lizard, scope_type="websocket")

    auth = WebSocketAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    with pytest.raises(WebSocketException) as exc_info:
        await auth(connection)

    assert exc_info.value.code == 1008
    assert exc_info.value.reason == message


async def test_http_gateway_allows_system_principal_with_real_hook(monkeypatch):
    async def fake_get_setting(agent_id, plugin_id):
        return {"management_active": True, "management_message": "Sistema in manutenzione"}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    admin = _make_admin_user()
    lizard = _make_lizard_with_real_hook(admin)
    connection = _make_connection(lizard, scope_type="http")

    auth = HTTPAuth(resource=AuthResource.CHAT, permission=AuthPermission.WRITE)
    result = await auth(connection)

    assert isinstance(result, AuthorizedInfo)
    assert result.user is admin


# ---------------------------------------------------------------------------
# (c) normal-mode / RITA read: GET /plugins/settings/mgmt_message
# ---------------------------------------------------------------------------

async def test_get_plugin_settings_normal_mode(secure_client, secure_client_headers, cheshire_cat):
    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": False,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    await _store(payload)

    # admin (system agent) reads the global settings — this is the RITA read path
    response = await secure_client.get("/plugins/system/settings/mgmt_message", headers=secure_client_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "mgmt_message"
    assert body["value"] == payload
    assert set(body["value"].keys()) == {
        "management_message",
        "management_active",
        "global_message",
        "show_global_msg",
    }
    # the schema exposes the same 4 fields
    assert set(body["scheme"]["properties"].keys()) == {
        "management_message",
        "management_active",
        "global_message",
        "show_global_msg",
    }

    await _cleanup()


# ---------------------------------------------------------------------------
# (d) management_active=false -> no-op
# ---------------------------------------------------------------------------

async def test_auth_request_noop_when_inactive(monkeypatch):
    async def fake_get_setting(agent_id, plugin_id):
        return {"management_active": False, "management_message": "Sistema in manutenzione"}

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    result = await auth_request.function(_make_user(), "local", None)

    assert result is None


async def test_auth_request_noop_when_no_setting(monkeypatch):
    async def fake_get_setting(agent_id, plugin_id):
        return None

    monkeypatch.setattr(crud_plugins, "get_setting", fake_get_setting)

    result = await auth_request.function(_make_user(), "local", None)

    assert result is None


async def test_public_global_message_endpoint_no_auth(client, secure_client, secure_client_headers, cheshire_cat):
    # activate the plugin so its custom endpoint is registered
    await secure_client.put("/plugins/toggle/mgmt_message", headers=secure_client_headers)

    payload = {
        "management_message": "Sistema in manutenzione",
        "management_active": False,
        "global_message": "Avviso globale",
        "show_global_msg": True,
    }
    await _store(payload)

    # unauthenticated client (no headers) must still reach the endpoint
    response = await client.get("/mgmt_message/global_message")

    assert response.status_code == 200
    assert response.json() == payload

    await _cleanup()