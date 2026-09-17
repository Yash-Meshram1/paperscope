import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import asdict
import requests

from paperscope.engine import Engine, Config, validate_chunks
from paperscope.generation import build_context
from test_retrieval import chunk, result


class EngineTests(unittest.TestCase):
    def test_duplicate_saved_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_chunks([chunk(), chunk()])

    def test_empty_skips_generation(self):
        engine = Engine(Config(Path('.')), [], None, None, None, None, {})
        with patch('paperscope.engine.generate_answer') as generate:
            self.assertIn('not contain enough',engine.answer('q',[]))
            generate.assert_not_called()

    def test_generation_receives_exact_evidence(self):
        evidence = [result('b'), result('a')]
        engine = Engine(Config(Path('.')), [], None, None, None, None, {})
        with patch('paperscope.engine.generate_answer',return_value='answer') as generate:
            engine.answer('q',evidence)
            self.assertIs(generate.call_args.args[1],evidence)
        context = build_context(evidence)
        self.assertLess(context.index('Paper: b'), context.index('Paper: a'))


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py'))

    def submit(self, question):
        self.app.text_area[0].set_value(question)
        self.app.button[0].click().run()

    def test_source_order_and_cached_initialization(self):
        evidence = [result('b'),result('a')]
        engine = Mock()
        engine.retrieve.return_value = evidence
        engine.answer.return_value = 'Answer [SOURCE 2]'
        with patch('paperscope.engine.load_engine',return_value=engine) as loader:
            self.app.run()
            self.submit('Compare DPR and ColBERT')
            self.assertFalse(self.app.exception)
            self.assertEqual([e.label for e in self.app.expander],['[SOURCE 1] b','[SOURCE 2] a'])
            self.assertIs(engine.answer.call_args.args[1],evidence)
            self.submit('What does DPR do?')
            self.assertEqual(loader.call_count,1)

    def test_empty_and_generation_failure(self):
        engine = Mock()
        engine.retrieve.return_value=[]
        engine.answer.return_value='Insufficient evidence'
        with patch('paperscope.engine.load_engine',return_value=engine):
            self.app.run(); self.submit('Unknown paper')
            self.assertEqual(len(self.app.expander),0)
            self.assertTrue(self.app.info)
            engine.retrieve.return_value=[result('dpr')]
            engine.answer.side_effect=requests.Timeout('timed out')
            self.submit('DPR')
            self.assertTrue(self.app.error)
            self.assertIn('[SOURCE 1]',self.app.expander[-1].label)
            self.assertFalse(self.app.exception)

    def test_initialization_failure_and_blank(self):
        with patch('paperscope.engine.load_engine',side_effect=RuntimeError('storage locked')) as loader:
            self.app.run(); self.submit(' ')
            loader.assert_not_called()
            self.submit('DPR')
            self.assertTrue(self.app.error)
            self.assertFalse(self.app.exception)


if __name__ == '__main__': unittest.main()
