import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "agenthn" / "webapp" / "static"
FIXTURES = STATIC / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


class CachedDemoTests(unittest.TestCase):
    def test_frontend_has_no_live_backend_paths(self):
        app = (STATIC / "app.js").read_text()
        index = (STATIC / "index.html").read_text()
        for forbidden in ("/api/", "EventSource", "trycloudflare.com", "isLive()", "BACKEND +"):
            self.assertNotIn(forbidden, app)
        self.assertNotIn("config.js", index)
        self.assertFalse((STATIC / "config.js").exists())
        self.assertRegex(app, r"const REPLAY_STEP_MS = 200;")
        self.assertIn("await sleep(REPLAY_STEP_MS);", app)

    def test_static_action_buttons_are_wired(self):
        app = (STATIC / "app.js").read_text()
        index = (STATIC / "index.html").read_text()
        button_ids = set(re.findall(r'<button[^>]+id="([^"]+)"', index))
        actions = {
            "m1Run", "m1Reset", "pResetAll", "pRepersonalize", "pReset2",
            "skRun", "skReset", "rcReset", "rcClassifyBtn",
            "rcInternalizeBtn", "rcAskAgainBtn", "rcQuestionsClose",
        }
        self.assertTrue(actions.issubset(button_ids))
        for button_id in actions:
            self.assertIn(f'$("{button_id}").onclick', app)

    def test_memory_inventory_matches_real_complete_files(self):
        meta = load("memory_meta.json")
        self.assertEqual(len(meta["scenarios"]), 3)
        for recording in meta["recordings"]:
            scenario, size = recording.split(":", 1)
            frames = load(f"memory_{scenario}_{size}.json")
            payloads = [item["f"] for item in frames]
            types = [frame["type"] for frame in payloads]
            self.assertEqual(types.count("meta"), 1)
            self.assertEqual(types[-1], "done")
            self.assertEqual(types.count("turn"), meta["turns_per_size"][size])
            self.assertGreater(types.count("query"), 0)

    def test_every_memory_file_is_in_inventory(self):
        meta = load("memory_meta.json")
        expected = set(meta["recordings"])
        actual = set()
        pattern = re.compile(r"memory_(.+)_(small|medium|large)\.json$")
        for path in FIXTURES.glob("memory_*.json"):
            match = pattern.match(path.name)
            if match:
                actual.add(f"{match.group(1)}:{match.group(2)}")
        self.assertEqual(actual, expected)

    def test_personalization_recordings_are_complete(self):
        fixture = load("personalization.json")
        self.assertEqual(len(fixture["observe"]), 4)
        self.assertEqual(len(fixture["chat"]), 4)
        self.assertTrue(fixture["repersonalize"])
        for response in fixture["observe"].values():
            self.assertTrue(response["reply"])
            self.assertIn("profile", response)
            self.assertIn("diff", response)
        for response in fixture["chat"].values():
            self.assertTrue(response["true"])
            self.assertTrue(response["false"])

    def test_skill_recordings_are_complete(self):
        frames = load("skills_product.json")
        types = [item["f"]["type"] for item in frames]
        self.assertEqual(types[0], "meta")
        self.assertEqual(types[-1], "done")
        self.assertEqual(types.count("round_start"), types.count("round_done"))
        router = load("skills_router.json")
        self.assertEqual(len(router), 2)
        required = {"converse", "classify", "internalize", "converse_again"}
        for recording in router.values():
            self.assertTrue(required.issubset(recording))


if __name__ == "__main__":
    unittest.main()
