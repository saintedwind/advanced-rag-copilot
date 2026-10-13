"""Pinned local SQuAD2 extractive reader; evidence rescue, not calibrated entailment."""

import hashlib
import threading
from pathlib import Path
import re
import numpy as np
import httpx
from src.rag.evidence import instruction_like, scope_documents

MODEL = "onnx-community/tinyroberta-squad2-ONNX"
REVISION = "7c9f69b7e6228375169a4553bcfa6639152e3a69"
WEIGHT_SHA256 = "4db70e7a019e32bf652fb9d30a5c24d18b6949fae2f1dd00d9fe6d43052b77cd"
TOKENIZER_SHA256 = "7b62d797c95d1563d3467f48daad0eca3e150f2d587e7e687b557c9ef1b25473"
PROTECTED = {
    "vendor",
    "salary",
    "password",
    "quota",
    "fee",
    "price",
    "bank",
    "account",
    "mobile",
    "application",
    "subscription",
    "escrow",
}


class SpanReader:
    def __init__(self, cache, enabled=True, margin=3.0):
        self.cache = Path(cache)
        self.enabled = enabled
        self.margin = margin
        self.lock = threading.RLock()
        self.session = None

    def load(self):
        if self.session is not None:
            return
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self.cache.mkdir(parents=True, exist_ok=True)
        for remote, local in [
            ("onnx/model_quantized.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
        ]:
            target = self.cache / local
            if not target.exists():
                temp = self.cache / (local + ".partial")
                url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{remote}?download=true"
                with httpx.stream(
                    "GET", url, follow_redirects=True, timeout=120
                ) as response:
                    response.raise_for_status()
                    with temp.open("wb") as file:
                        total = 0
                        for part in response.iter_bytes():
                            total += len(part)
                            if total > 100_000_000:
                                raise ValueError("Reader download size exceeded")
                            file.write(part)
                temp.replace(target)
        if (
            hashlib.sha256((self.cache / "model.onnx").read_bytes()).hexdigest()
            != WEIGHT_SHA256
        ):
            raise ValueError("Reader model checksum mismatch")
        if (
            hashlib.sha256((self.cache / "tokenizer.json").read_bytes()).hexdigest()
            != TOKENIZER_SHA256
        ):
            raise ValueError("Reader tokenizer checksum mismatch")
        self.tokenizer = Tokenizer.from_file(str(self.cache / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=384, strategy="only_second")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.cache / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def score(self, question, context):
        encoded = self.tokenizer.encode(question, context)
        available = {
            i
            for i, (sequence, offset) in enumerate(
                zip(encoded.sequence_ids, encoded.offsets)
            )
            if sequence == 1 and offset[1] > offset[0]
        }
        inputs = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
        }
        if any(i.name == "token_type_ids" for i in self.session.get_inputs()):
            inputs["token_type_ids"] = np.asarray([encoded.type_ids], dtype=np.int64)
        start, end = self.session.run(None, inputs)
        start, end = start[0], end[0]
        null = float(start[0] + end[0])
        best = None
        for i in available:
            for j in range(i, min(i + 48, len(encoded.ids))):
                if j not in available:
                    continue
                score = float(start[i] + end[j]) - null
                if best is None or score > best["margin"]:
                    a, b = encoded.offsets[i][0], encoded.offsets[j][1]
                    best = {
                        "margin": score,
                        "start_char": a,
                        "end_char": b,
                        "text": context[a:b],
                    }
        return best

    def rescue(self, question, docs, screen):
        if not self.enabled or PROTECTED.intersection(screen.get("missing_terms", [])):
            return [], {"status": "disabled_or_protected_facet"}
        flags = set(re.findall(r"--[a-z][a-z0-9-]*", question.lower()))
        safe = [
            d
            for d in scope_documents(question, docs)[:3]
            if not instruction_like(d.page_content)
            and all(
                flag in d.page_content.lower()
                or flag in d.metadata.get("document_identifiers", [])
                for flag in flags
            )
        ]
        if not safe:
            return [], {"status": "no_safe_candidates"}
        evidence = []
        try:
            with self.lock:
                self.load()
                for doc in safe:
                    prediction = self.score(question, doc.page_content)
                    valid = (
                        prediction
                        and type(prediction["start_char"]) is int
                        and type(prediction["end_char"]) is int
                        and 0
                        <= prediction["start_char"]
                        < prediction["end_char"]
                        <= len(doc.page_content)
                        and doc.page_content[
                            prediction["start_char"] : prediction["end_char"]
                        ]
                        == prediction["text"]
                        and np.isfinite(prediction["margin"])
                    )
                    if valid and prediction["margin"] >= self.margin:
                        evidence.append((doc, prediction))
        except Exception:
            return [], {
                "status": "reader_unavailable",
                "fallback": "conservative_screen",
            }
        if not evidence:
            return [], {"status": "no_answer_span", "margin_threshold": self.margin}
        doc, prediction = max(evidence, key=lambda item: item[1]["margin"])
        prediction["chunk_id"] = doc.metadata["chunk_id"]
        return [doc], {
            "status": "span_evidence_found",
            "model": MODEL,
            "revision": REVISION,
            "margin_threshold": self.margin,
            "prediction": prediction,
            "not_calibrated_confidence": True,
        }
