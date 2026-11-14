"""Offline integration and regression tests. Run: python -m unittest -v"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import main
import tiktoken

ROOT = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'


def data_for(text, ids, challenge):
    return {'calibration': {'text': text, 'token_ids': ids},
            'challenge': {'token_ids': challenge}}


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = patch.object(main, 'ROOT', self.root)
        self.patch.start()
        self.configure()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def configure(self, **kwargs):
        config = {'huggingface': [], 'sentencepiece': [], 'tokenizers': []}
        config.update(kwargs)
        (self.root / 'tokenizer_candidates.json').write_text(json.dumps(config), encoding='utf-8')

    def test_public_sample(self):
        source = json.loads((ROOT / 'train-input.json').read_text(encoding='utf-8'))
        expected = json.loads((ROOT / 'train-output.json').read_text(encoding='utf-8'))
        self.assertEqual(main.identify(source), expected)

    def test_official_test_dataset(self):
        source = json.loads((ROOT / 'test-input.json').read_text(encoding='utf-8'))
        expected = {
            'tokenizer_id': 'cl100k_base',
            'decoded_message': 'SIGMOID_TOKENIZATION\nCongratulations! You have successfully solved the American Tokenizer Mystery.\n    Verification Hash: 2026-USA-NLP-BPE\n        Cluster Node: Grand-Canyon-Compute-01\nSubword token IDs are arbitrary vocabulary matrix indices, not universal ASCII codes!'
        }
        self.assertEqual(main.identify(source), expected)

    def test_tiktoken_families_unicode_whitespace(self):
        text = '# America România 😀\ndef f(x):\n    return "New York" + str(x-24)\n     café\n'
        message = 'SIGMOID_EXAMPLE\r\nHello, România 😀!  \n'
        for name in ('p50k_base', 'r50k_base', 'cl100k_base', 'o200k_base'):
            with self.subTest(name=name):
                enc = tiktoken.get_encoding(name)
                result = main.identify(data_for(text, enc.encode(text), enc.encode(message)))
                self.assertEqual(result, {'tokenizer_id': name, 'decoded_message': message})

    def test_literal_special_token_can_be_ordinary_text(self):
        enc = tiktoken.get_encoding('cl100k_base')
        text = 'Literal <|endoftext|> in American text. România 😀'
        self.assertEqual(main.identify(data_for(text, enc.encode_ordinary(text), enc.encode('SIGMOID_EXAMPLE')))['decoded_message'], 'SIGMOID_EXAMPLE')

    def test_special_tokens_encoded_as_special(self):
        enc = tiktoken.get_encoding('p50k_edit')
        text = '<|fim_prefix|>def f():\n    pass<|fim_suffix|>!<|fim_middle|>'
        result = main.identify(data_for(text, enc.encode(text, allowed_special='all'), enc.encode('SIGMOID_EXAMPLE')))
        self.assertEqual(result['tokenizer_id'], 'p50k_edit')

    def test_invalid_ids_rejected(self):
        for invalid in ([-1], [True], [2.5], '123'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                main.identify(data_for('abc', invalid, []))

    def test_unknown_calibration_fails(self):
        with self.assertRaises(RuntimeError):
            main.identify(data_for('abc', [99999999], []))

    def test_raw_tokenizer_json(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        tok = Tokenizer(models.WordPiece({'[UNK]': 0, 'hello': 1, 'world': 2, '!': 3}, unk_token='[UNK]'))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.decoder = decoders.WordPiece(cleanup=False)
        path = self.root / 'local.json'
        tok.save(str(path))
        self.configure(tokenizers=['local.json'])
        result = main.identify(data_for('hello world!', tok.encode('hello world!').ids, tok.encode('hello!').ids))
        self.assertEqual(result, {'tokenizer_id': 'local', 'decoded_message': tok.decode(tok.encode('hello!').ids, skip_special_tokens=False)})

    def test_huggingface_local_with_special_tokens(self):
        from transformers import BertTokenizer
        location = self.root / 'local_bert'
        location.mkdir()
        (location / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n!\n', encoding='utf-8')
        tok = BertTokenizer(vocab_file=str(location/'vocab.txt'), do_lower_case=False)
        tok.save_pretrained(str(location))
        self.configure(huggingface=['local_bert'])
        result = main.identify(data_for('hello world!', tok.encode('hello world!', add_special_tokens=True), tok.encode('hello!', add_special_tokens=False)))
        self.assertEqual(result, {'tokenizer_id': 'local_bert', 'decoded_message': tok.decode(tok.encode('hello!', add_special_tokens=False), clean_up_tokenization_spaces=False)})

    def test_ambiguous_calibration_does_not_guess(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        for name, word in (('first', 'alpha'), ('second', 'beta')):
            tok = Tokenizer(models.WordLevel({'[UNK]': 0, 'hello': 1, 'world': 2, word: 3}, unk_token='[UNK]'))
            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tok.save(str(self.root / (name + '.json')))
        self.configure(tokenizers=['first.json', 'second.json'])
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous calibration'):
            main.identify(data_for('hello world', [1, 2], [3]))

    def test_sentencepiece_boundary_tokens(self):
        import sentencepiece as spm
        corpus = self.root / 'corpus.txt'
        corpus.write_text(('hello world\nAmerican calibration text\nmessage example\n') * 20, encoding='utf-8')
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(self.root/'tiny'), vocab_size=24, hard_vocab_limit=False, minloglevel=2)
        tok = spm.SentencePieceProcessor(model_file=str(self.root/'tiny.model'))
        self.configure(sentencepiece=['tiny.model'])
        for bos, eos in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(bos=bos, eos=eos):
                result = main.identify(data_for('hello world', tok.encode('hello world', add_bos=bos, add_eos=eos), tok.encode('message example')))
                self.assertEqual(result, {'tokenizer_id': 'tiny', 'decoded_message': 'message example'})

    def test_cli_defaults_use_current_directory(self):
        enc = tiktoken.get_encoding('cl100k_base')
        source = data_for('Calibration: România 😀  San Francisco\n', enc.encode('Calibration: România 😀  San Francisco\n'), enc.encode('SIGMOID_EXAMPLE'))
        (self.root/'test-input.json').write_text(json.dumps(source), encoding='utf-8')
        run = subprocess.run([sys.executable, str(ROOT/'main.py')], cwd=self.root, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads((self.root/'test-output.json').read_text()), {'tokenizer_id': 'cl100k_base', 'decoded_message': 'SIGMOID_EXAMPLE'})


if __name__ == '__main__':
    unittest.main()
