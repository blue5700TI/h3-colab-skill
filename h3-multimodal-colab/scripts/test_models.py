"""Offline regression tests for H3 model selection (no GPU or downloads)."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import h3local as L

R = L.R
G = R.G


def names(models):
    return {Path(name).name for _, name, _ in models}


def workflow_weights(graph):
    return {Path(value).name for node in graph.values() if isinstance(node, dict)
            for value in node.get("inputs", {}).values()
            if isinstance(value, str) and value.endswith(".safetensors")}


class ModelSelectionTests(unittest.TestCase):
    def test_image_only_matches_active_image_workflow(self):
        expected = workflow_weights(G.load_template(G.PICTURE_TEMPLATE))
        self.assertEqual(len(expected), 3)
        for pictures in (False, True):
            with self.subTest(pictures=pictures):
                selected = R.models_for([{"type": "image"}], pictures)
                self.assertEqual(names(selected), expected)
                self.assertEqual(len(selected), len(expected))

    def test_official_with_image_does_not_add_community_pack(self):
        video = {"type": "video", "mode": "text", "model_set": "official"}
        selected = R.models_for([video, {"type": "image"}])
        self.assertEqual(names(selected), names(R.SHARED_ENCODER + R.COMMON_MODELS
                                                + R.OFFICIAL_MODELS["ref"] + R.PICTURE_MODELS))

    def test_official_with_picture_flag_does_not_add_community_pack(self):
        video = {"type": "video", "mode": "frames", "model_set": "official"}
        selected = R.models_for([video], pictures=True)
        self.assertEqual(names(selected), names(R.SHARED_ENCODER + R.COMMON_MODELS
                                                + R.OFFICIAL_MODELS["fl"] + R.PICTURE_MODELS))

    def test_official_first_last_alias_uses_fl_weights(self):
        selected = R.models_for([{"type": "video", "mode": "first_last", "model_set": "official"}])
        self.assertEqual(names(selected), names(R.SHARED_ENCODER + R.COMMON_MODELS + R.OFFICIAL_MODELS["fl"]))

    def test_default_video_keeps_template_weights_and_adds_requested_anime_vae(self):
        selected = R.models_for([{"type": "video", "vae": "AnImE"}])
        expected = workflow_weights(G.load_template(G.VIDEO_TEMPLATE)) | {G.ANIME_VIDEO_VAE}
        self.assertTrue(expected <= names(selected), expected - names(selected))
        unmodified = R.models_for([{"type": "video"}])
        self.assertNotIn(G.ANIME_VIDEO_VAE, names(unmodified))

    def test_tts_keeps_template_weights_even_in_official_batch(self):
        selected = R.models_for([{"type": "tts"}, {"type": "video", "model_set": "official"}])
        expected = workflow_weights(G.load_template("h3_tts32_api.json"))
        self.assertTrue(expected <= names(selected), expected - names(selected))
        self.assertTrue(names(R.OFFICIAL_MODELS["ref"]) <= names(selected))


class DesktopManifestTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.manifest = self.base / "manifest.json"
        self.enterContext(patch.object(L, "get", side_effect=self.comfy_response))
        self.fetch = self.enterContext(patch.object(L, "fetch"))
        self.enterContext(patch.object(L, "git_clone", side_effect=AssertionError("unexpected clone")))

    def comfy_response(self, url, timeout=5):
        if url.endswith("/system_stats"):
            return {"system": {"argv": [str(self.base / "main.py")]}}
        if url.endswith("/object_info"):
            tts = G.load_template("h3_tts32_api.json")
            return set(G.required_classes(True)) | {n["class_type"] for n in tts.values()}
        if "/models/" in url:
            return []
        raise AssertionError(f"unexpected request: {url}")

    def invoke(self, *args):
        out = io.StringIO()
        argv = ["h3local.py", "--models-dir", str(self.base / "models"), *args]
        with patch.object(sys, "argv", argv), redirect_stdout(out):
            try:
                L.main()
            except SystemExit as e:
                return e.code, out.getvalue()
        return 0, out.getvalue()

    def assert_fetches(self, expected):
        self.assertEqual([c.args for c in self.fetch.call_args_list],
                         [(repo, f, self.base / "models" / folder / Path(f).name) for repo, f, folder in expected])

    def test_setup_downloads_only_models_from_image_manifest(self):
        self.manifest.write_text(json.dumps({"jobs": [{"id": "pic", "type": "image", "prompt": "A cat"}]}),
                                 encoding="utf-8")
        code, output = self.invoke("setup", "--manifest", str(self.manifest))
        self.assertEqual(code, 0, output)
        self.assert_fetches(R.models_for([{"type": "image"}]))

    def test_image_only_does_not_require_video_or_tts_nodes(self):
        self.manifest.write_text(json.dumps({"jobs": [{"id": "pic", "type": "image", "prompt": "A cat"}]}),
                                 encoding="utf-8")
        with patch.object(L, "get", side_effect=self.image_only_response):
            code, output = self.invoke("check", "--manifest", str(self.manifest))
        self.assertEqual(code, 1, output)
        self.assertNotIn("ComfyUI 不認得這些節點", output)
        self.assertIn("少 ", output)

    def image_only_response(self, url, timeout=5):
        if url.endswith("/object_info"):
            return set(G.required_classes(True, kinds=("image",)))
        return self.comfy_response(url, timeout)

    def test_setup_check_and_run_use_same_manifest_models_including_anime_vae(self):
        video = {"id": "anime", "type": "video", "mode": "text", "prompt": "A cat", "vae": "anime"}
        self.manifest.write_text(json.dumps({"model_set": "official", "install_pictures": True, "jobs": [video]}),
                                 encoding="utf-8")
        expected = R.models_for([dict(video, model_set="official")], pictures=True)
        expected_line = f"❌ 少 {len(expected)} 個模型：" + "、".join(Path(f).name for _, f, _ in expected)
        self.assertIn(G.ANIME_VIDEO_VAE, expected_line)
        for args in (("check", "--manifest", str(self.manifest)), ("run", str(self.manifest))):
            with self.subTest(command=args[0]):
                code, output = self.invoke(*args)
                self.assertEqual(code, 1, output)
                self.assertIn(expected_line, output.splitlines())
        self.fetch.assert_not_called()
        code, output = self.invoke("setup", "--manifest", str(self.manifest))
        self.assertEqual(code, 0, output)
        self.assert_fetches(expected)


if __name__ == "__main__":
    unittest.main()
