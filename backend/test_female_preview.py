"""Exercise preview transfer guards without imports, downloads or a cloud runtime."""
import ast
import io
from pathlib import Path
import runpy
import unittest
from unittest.mock import Mock, patch
import urllib.parse
import urllib.request

SOURCE = Path(__file__).parent / "voice_research/female_preview.py"


def transport():
    tree = ast.parse(SOURCE.read_text())
    names = {"HOSTS", "MIB", "received", "requests"}
    nodes = [n for n in tree.body if
             isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {"checked", "Redirects", "fetch"}
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    namespace = {"urllib": urllib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "preview-transport", "exec"), namespace)
    namespace["opener"] = Mock()
    return namespace


class FemalePreviewTests(unittest.TestCase):
    def test_only_approved_https_hosts_include_observed_cdn(self):
        n = transport()
        url = "https://us.gcp.cdn.hf.co/asset.mp3"
        self.assertEqual(n["checked"](url), url)
        for url in ("https://evil.example/asset.mp3", "http://huggingface.co/file", "https://user:secret@huggingface.co/file"):
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                n["checked"](url)

    def test_declared_large_file_is_refused_before_body_read(self):
        n = transport()
        response = Mock()
        response.headers = {"Content-Length": "1025"}
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        n["opener"].open.return_value = response
        with self.assertRaises(RuntimeError):
            n["fetch"]("https://huggingface.co/example.mp3", 1024)
        response.read.assert_not_called()

    def test_shared_budget_and_request_limit_stop_further_fetches(self):
        n = transport()
        class Response(io.BytesIO):
            headers = {}
        n["opener"].open.return_value = Response(b"xxxx")
        n["received"] = 3*n["MIB"] - 3
        with self.assertRaises(RuntimeError):
            n["fetch"]("https://huggingface.co/example.mp3", 1024)
        self.assertEqual(n["received"], 3*n["MIB"])
        with self.assertRaises(RuntimeError):
            n["fetch"]("https://huggingface.co/example.mp3", 1024)
        self.assertEqual(n["opener"].open.call_count, 1)
        n = transport()
        n["requests"] = 8
        with self.assertRaises(RuntimeError):
            n["fetch"]("https://huggingface.co/example.mp3", 1024)
        n["opener"].open.assert_not_called()

    def test_mac_stops_before_optional_imports_or_network(self):
        with patch("platform.system", return_value="Darwin"), \
                patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Network on Mac")):
            with self.assertRaisesRegex(RuntimeError, "hosted Colab"):
                runpy.run_path(str(SOURCE), run_name="__main__")


if __name__ == "__main__":
    unittest.main()
