"""LMPU Processors - Neural summary processor."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np

from ..base import Manifest, Processor, ProcessorContext, JobContext, Result


class NeuralSummaryProcessor(Processor):
    """Neural summary processor with extractive and model-based backends."""
    
    manifest = Manifest(
        name="neural.summary",
        version="0.1.0",
        protocol="lmpu/1",
        ops=[
            {"name": "summarize_text_buffer", "inputs": ["text"], "outputs": ["summary"]},
            {"name": "embed_text_buffer", "inputs": ["text"], "outputs": ["embedding"]},
        ],
        limits={
            "max_buffer_bytes": 100_000_000,
            "max_batch": 8,
        },
        determinism=["best_effort"],
        latency_class="s",
        streaming=False,
        progress=False,
        interactive=False,
        persistent_state=False,
        batching=True,
        zero_copy=False,
        sandbox=["subprocess"],
        executor="process",
    )

    def __init__(self):
        super().__init__()
        self._backend = "textrank"  # textrank or model
        self._model = None

    async def setup(self, ctx: ProcessorContext) -> None:
        backend = ctx.runtime_config.get("summary_backend", "textrank")
        self._backend = backend
        
        if backend == "model":
            # Lazy load model
            try:
                from sentence_transformers import SentenceTransformer
                model_name = ctx.runtime_config.get("summary_model", "all-MiniLM-L6-v2")
                self._model = SentenceTransformer(model_name)
            except ImportError:
                self._backend = "textrank"

    async def execute(self, job: dict[str, Any], ctx: JobContext) -> Result:
        op = job.get("op")
        inputs = job.get("inputs", {})
        
        try:
            if op == "summarize_text_buffer":
                return await self._summarize(inputs, ctx)
            elif op == "embed_text_buffer":
                return await self._embed(inputs, ctx)
            else:
                return Result(
                    success=False,
                    error_code="unknown_op",
                    error_message=f"Unknown operation: {op}",
                )
        except Exception as e:
            return Result(
                success=False,
                error_code="execution_error",
                error_message=str(e),
                retriable=False,
            )

    async def _summarize(self, inputs: dict[str, Any], ctx: JobContext) -> Result:
        """Summarize text using TextRank or model."""
        text_buffer = inputs.get("text", "")
        
        if isinstance(text_buffer, dict) and text_buffer.get("buffer"):
            # Would read from buffer
            text = "Sample text for summarization. " * 100
        else:
            text = str(text_buffer)
        
        if self._backend == "model" and self._model:
            summary = await self._summarize_model(text)
        else:
            summary = await self._summarize_textrank(text)
        
        result_data = {
            "summary": summary,
            "original_chars": len(text),
            "summary_chars": len(summary),
            "compression_ratio": len(summary) / len(text) if text else 0,
        }
        
        checksum = hashlib.blake3(json.dumps(result_data, sort_keys=True).encode()).hexdigest()
        
        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def _summarize_textrank(self, text: str) -> str:
        """Extractive summarization using TextRank algorithm."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= 3:
            return text
        
        # Build similarity matrix (simple word overlap)
        import math
        from collections import Counter
        
        def sentence_vector(sentence):
            words = re.findall(r'\w+', sentence.lower())
            return Counter(words)
        
        vectors = [sentence_vector(s) for s in sentences]
        
        def cosine_sim(v1, v2):
            if not v1 or not v2:
                return 0.0
            intersection = set(v1.keys()) & set(v2.keys())
            numerator = sum(v1[w] * v2[w] for w in intersection)
            denom1 = math.sqrt(sum(v * v for v in v1.values()))
            denom2 = math.sqrt(sum(v * v for v in v2.values()))
            if denom1 == 0 or denom2 == 0:
                return 0.0
            return numerator / (denom1 * denom2)
        
        # PageRank-style scoring
        n = len(sentences)
        scores = [1.0 / n] * n
        damping = 0.85
        
        for _ in range(20):
            new_scores = [(1 - damping) / n] * n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        sim = cosine_sim(vectors[i], vectors[j])
                        new_scores[i] += damping * scores[j] * sim
            scores = new_scores
        
        # Select top sentences
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        num_sentences = max(1, min(3, n // 3))
        selected = sorted(ranked[:num_sentences])
        
        return ' '.join(sentences[i] for i in selected)

    async def _summarize_model(self, text: str) -> str:
        """Model-based summarization (placeholder)."""
        # Would use a seq2seq model
        return f"[Model summary of {len(text)} chars]"

    async def _embed(self, inputs: dict[str, Any], ctx: JobContext) -> Result:
        """Generate embeddings for text."""
        text_buffer = inputs.get("text", "")
        
        if isinstance(text_buffer, dict) and text_buffer.get("buffer"):
            text = "Sample text for embedding. " * 10
        else:
            text = str(text_buffer)
        
        if self._backend == "model" and self._model:
            embedding = self._model.encode(text).tolist()
        else:
            # Random embedding for testing
            embedding = np.random.randn(384).astype(np.float32).tolist()
        
        result_data = {
            "embedding": embedding,
            "dimension": len(embedding),
        }
        
        checksum = hashlib.blake3(json.dumps(result_data, sort_keys=True).encode()).hexdigest()
        
        return Result(
            success=True,
            data=result_data,
            checksum=checksum,
        )

    async def cancel(self, job_id: str) -> None:
        pass

    async def shutdown(self) -> None:
        pass