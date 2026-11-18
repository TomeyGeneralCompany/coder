"""Native-only profile persistence contracts after hosted provider removal."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openhands.sdk.llm import LLM
from openhands.sdk.llm.llm_profile_store import LLMProfileStore
from openhands.sdk.errors import Invalid
from openhands.sdk.settings.model import OpenHandsAgentSettings


class NativeProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.store = LLMProfileStore(self.path)

    def test_round_trip_native_profile_without_api_key(self):
        model = LLM(
            model="qwen3:4b",
            max_input_tokens=16384,
            max_output_tokens=2048,
            base_url="http://127.0.0.1:11434",
        )
        self.store.save("coding", model)
        persisted = json.loads((self.path / "coding.json").read_text())
        self.assertNotIn("api_key", persisted)
        self.assertNotIn("provider_connection_id", persisted)
        self.assertEqual(self.store.load("coding").max_input_tokens, 16384)
        self.assertEqual(self.store.list_summaries()[0]["api_key_set"], False)

    def test_legacy_hosted_profile_refused_without_mutation(self):
        legacy = {"model": "gpt-4o", "api_key": "encrypted-secret", "base_url": "https://example.invalid"}
        source = self.path / "old.json"
        source.write_text(json.dumps(legacy))
        before = source.read_bytes()
        with self.assertRaises((ValueError, Invalid)):
            self.store.load("old")
        self.assertEqual(source.read_bytes(), before)
        summaries = self.store.list_summaries()
        self.assertEqual(summaries[0]["model"], "gpt-4o")
        self.assertTrue(summaries[0]["provider_connection_broken"])
        self.assertFalse(summaries[0]["api_key_set"])

    def test_legacy_connection_ref_refused_without_live_credentials(self):
        path = self.path / "linked.json"
        path.write_text(json.dumps({"model": "qwen3:4b", "provider_connection_id": "cloud", "base_url": "http://127.0.0.1:11434"}))
        with self.assertRaises((ValueError, Invalid)):
            self.store.load("linked")

    def test_settings_create_agent_uses_local_model_directly(self):
        settings = OpenHandsAgentSettings(llm=LLM())
        agent = settings.create_agent()
        self.assertEqual(agent.llm.model, "qwen3:4b")
        self.assertEqual(agent.llm.base_url, "http://127.0.0.1:11434")

    def test_settings_forbid_hosted_model_router(self):
        settings = OpenHandsAgentSettings(
            llm=LLM(), enable_classify_and_switch_llm_tool=True
        )
        with self.assertRaisesRegex(ValueError, "routing between LLM profiles"):
            settings.create_agent()

    def test_local_profile_server_preflight_does_not_need_subscription(self):
        from openhands.agent_server.profiles_router import _preflight_llm
        llm = _preflight_llm(LLM(max_input_tokens=8192))
        self.assertEqual(llm.model, "qwen3:4b")
        self.assertEqual(llm.num_retries, 0)

    def test_agent_server_import_and_no_provider_credential_routes(self):
        from openhands.agent_server.api import create_app
        # Building the ASGI graph must not import deleted provider modules.
        app = create_app()
        paths = [str(route.path) for route in app.routes]
        self.assertFalse(any("provider-connections" in path for path in paths))
        self.assertTrue(any(path.startswith("/api/llm") for path in paths))

    def test_profile_save_delete_and_rename(self):
        self.store.save("alpha", LLM())
        self.store.rename("alpha", "beta")
        self.assertTrue((self.path / "beta.json").exists())
        self.assertEqual(self.store.load("beta").model, "qwen3:4b")
        self.store.delete("beta")
        self.assertFalse((self.path / "beta.json").exists())


if __name__ == "__main__":
    unittest.main()
