"""Identify a tokenizer by exact calibration, then decode challenge tokens."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
# Bundled tiktoken vocabularies allow evaluation without network access.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "tiktoken_cache"))


def identify(data):
    text = data["calibration"]["text"]
    expected = data["calibration"]["token_ids"]
    challenge = data["challenge"]["token_ids"]
    if not isinstance(text, str):
        raise ValueError("calibration.text must be a string")
    for ids in (expected, challenge):
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("token_ids must be lists of nonnegative integers")

    matches, errors = [], []

    def attempt(name, encode, decode):
        try:
            if encode(text) == expected:
                matches.append((name, decode(challenge)))
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    try:
        import tiktoken
        # Prefer base encodings to their special-token extensions when equivalent.
        preferred = ["p50k_base", "r50k_base", "cl100k_base", "o200k_base", "gpt2", "p50k_edit"]
        names = list(dict.fromkeys(preferred + tiktoken.list_encoding_names()))
        for name in names:
            try:
                enc = tiktoken.get_encoding(name)
                attempt(name, lambda s, e=enc: e.encode(s, allowed_special="all"), enc.decode)
                # Literal special-token strings may have been encoded as ordinary text.
                if any(token in text for token in enc.special_tokens_set):
                    attempt(name, enc.encode_ordinary, enc.decode)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    except ImportError:
        errors.append("Install tiktoken from requirements.txt")

    # A finite calibration cannot distinguish every possible custom vocabulary.
    # Additional known candidates may be registered explicitly.
    config_path = ROOT / "tokenizer_candidates.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    # Try optional families only if tiktoken did not yield a complete match.
    if not matches or any(config.get(key) for key in ("huggingface", "sentencepiece", "tokenizers")):
        candidates = config.get("huggingface", [
            "openai-community/gpt2", "FacebookAI/roberta-base",
            "google-bert/bert-base-uncased", "google-bert/bert-base-cased",
            "google-t5/t5-small", "FacebookAI/xlm-roberta-base",
            "EleutherAI/gpt-neo-125m", "EleutherAI/gpt-neox-20b",
            "bigscience/bloom-560m", "mistralai/Mistral-7B-v0.1",
        ])
        try:
            from transformers import AutoTokenizer
            for name in candidates:
                try:
                    location = ROOT / name
                    source = str(location) if location.exists() else name
                    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
                    for special in (False, True):
                        attempt(name,
                                lambda s, t=tok, a=special: t.encode(s, add_special_tokens=a),
                                lambda ids, t=tok: t.decode(ids, skip_special_tokens=False,
                                                           clean_up_tokenization_spaces=False))
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
        except ImportError:
            errors.append("Optional fallback requires transformers and sentencepiece")

        try:
            from tokenizers import Tokenizer
            paths = list((ROOT / "tokenizers").glob("*.json"))
            paths.extend(ROOT / entry for entry in config.get("tokenizers", []))
            for path in dict.fromkeys(paths):
                try:
                    tok = Tokenizer.from_file(str(path))
                    for special in (False, True):
                        attempt(path.stem,
                                lambda s, t=tok, a=special: t.encode(s, add_special_tokens=a).ids,
                                lambda ids, t=tok: t.decode(ids, skip_special_tokens=False))
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
        except ImportError:
            errors.append("Loading tokenizer.json requires tokenizers")

        try:
            import sentencepiece as spm
            models = list((ROOT / "tokenizers").glob("*.model"))
            for entry in config.get("sentencepiece", []):
                models.append(ROOT / entry)
            for path in dict.fromkeys(models):
                try:
                    tok = spm.SentencePieceProcessor(model_file=str(path))
                    for bos, eos in ((False, False), (True, False), (False, True), (True, True)):
                        attempt(path.stem,
                                lambda s, t=tok, b=bos, e=eos: t.encode(
                                    s, out_type=int, add_bos=b, add_eos=e), tok.decode)
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
        except ImportError:
            pass

    if not matches:
        raise RuntimeError("No exact calibration match. Add the required vocabulary to "
                           "tokenizer_candidates.json or tokenizers/.\n" + "\n".join(errors))
    matches = list(dict.fromkeys(matches))
    messages = {message for _, message in matches}
    if len(messages) != 1:
        raise RuntimeError("Ambiguous calibration: matching tokenizers decode the challenge "
                           "differently: " + ", ".join(name for name, _ in matches))
    if len(matches) > 1:
        print("Equivalent candidates: " + ", ".join(name for name, _ in matches), file=sys.stderr)
    name, message = matches[0]
    return {"tokenizer_id": name, "decoded_message": message}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="test-input.json")
    parser.add_argument("--output", default="test-output.json")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = identify(data)
    output = Path(args.output)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
