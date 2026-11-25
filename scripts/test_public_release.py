"""Exercise credential and private-output guards using synthetic strings only."""
from pathlib import Path
import unittest
from check_public_release import inspect_file


class ReleaseGuardTests(unittest.TestCase):
    def rules(self,path,body):
        return {x['rule'] for x in inspect_file(Path(path),body.encode())}

    def test_detects_credentials_without_returning_the_matched_value(self):
        value='sk-'+'x'*40
        issues=inspect_file(Path('example.py'),('key = '+value).encode())
        self.assertIn('openai-key',{x['rule'] for x in issues})
        self.assertNotIn(value,str(issues))
        self.assertIn('huggingface-token',self.rules('example.py','hf_'+'x'*30))
        self.assertIn('github-token',self.rules('example.py','ghp_'+'x'*40))

    def test_rejects_private_paths_and_google_document_links(self):
        self.assertIn('personal-macos-path',self.rules('example.py','/Users/'+'fixture'+'/Documents'))
        url='https://'+'drive.google.com/file/d/'+'fixture'
        self.assertIn('private-google-document',self.rules('example.py',url))
        self.assertIn('private-or-generated-file',self.rules('speech.wav','fixture'))
        self.assertIn('environment-file',self.rules('backend/.env','TOKEN=fixture'))

    def test_accepts_empty_environment_example_and_public_source_links(self):
        self.assertEqual(self.rules('backend/.env.example','ANYANWU_TEXT_KEY=\n'),set())
        self.assertEqual(self.rules('notice.md','https://huggingface.co/models'),set())

    def test_notebook_outputs_are_not_publishable(self):
        body='{"cells":[{"cell_type":"code","outputs":[{"text":"fixture"}],"execution_count":1}]}'
        self.assertIn('notebook-has-execution-output',self.rules('example.ipynb',body))


if __name__=='__main__':unittest.main()
