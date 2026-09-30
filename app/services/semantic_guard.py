"""Semantic Guard Observation Mode v1 - non-intrusive monitoring."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger("semantic_guard")

_REPORT_ALLOWLIST = {
    "imdb_id",
    "target_id_hash",
    "fingerprint",
    "anchor_count",
    "confidence",
    "confidence_reason",
    "slope",
    "offset_seconds",
    "inlier_count",
    "inlier_ratio",
    "coverage",
    "cv_median_seconds",
    "cv_p90_seconds",
    "alass_median_seconds",
    "alass_p90_seconds",
    "semantic_median_seconds",
    "disagreement_count",
    "proposed_corrections_count",
    "proposed_corrections",
    "served_result",
    "utc_timestamp",
    "model_identifier",
}


def _validate_imdb_id(imdb_id: str) -> str:
    if not re.fullmatch(r"tt\d{7,10}", imdb_id or ""):
        return "unknown"
    return imdb_id


def _sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return safe or "unknown"


def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _fingerprint(target_bytes: bytes, reference_bytes: bytes, alass_bytes: bytes, model_id: str) -> str:
    h = hashlib.sha256()
    h.update(target_bytes)
    h.update(reference_bytes)
    h.update(alass_bytes)
    h.update(model_id.encode("utf-8"))
    return h.hexdigest()


def _allowlisted_report(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k in _REPORT_ALLOWLIST}


@dataclass
class Observation:
    target_bytes: bytes
    reference_bytes: bytes
    alass_bytes: bytes
    meta: dict[str, Any]
    target_id: str
    imdb_id: str


class SemanticGuardObserver:
    """Observation-only semantic guard with lazy heavy imports."""

    def __init__(self) -> None:
        self.mode = settings.SEMANTIC_GUARD_MODE.lower()
        self.model_id = settings.SEMANTIC_GUARD_MODEL
        self.max_concurrency = 1
        self.queue_size = max(1, settings.SEMANTIC_GUARD_QUEUE_SIZE)
        self.timeout_seconds = max(1.0, settings.SEMANTIC_GUARD_TIMEOUT_SECONDS)
        self.report_dir = Path(settings.SEMANTIC_GUARD_REPORT_DIR)

        self._queue: asyncio.Queue[Observation | None] = asyncio.Queue(maxsize=self.queue_size)
        self._seen_fingerprints: set[str] = set()
        self._worker_task: asyncio.Task | None = None
        self._encoder: SentenceTransformer | None = None
        self._model_loaded = False
        self._model_load_failed = False
        self._closing = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.mode != "observe" or self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker())

    async def close(self) -> None:
        self._closing = True
        await self._queue.put(None)
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except TimeoutError:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except Exception:
                    pass
        self._worker_task = None

    def enqueue(self, target_bytes: bytes, reference_bytes: bytes, alass_bytes: bytes,
                meta: dict[str, Any], target_id: str) -> None:
        if self.mode != "observe":
            return

        imdb_id = _validate_imdb_id(str(meta.get("imdb_id", "")))
        fingerprint = _fingerprint(target_bytes, reference_bytes, alass_bytes, self.model_id)

        if fingerprint in self._seen_fingerprints:
            return

        obs = Observation(
            target_bytes=target_bytes,
            reference_bytes=reference_bytes,
            alass_bytes=alass_bytes,
            meta=meta,
            target_id=target_id,
            imdb_id=imdb_id,
        )

        try:
            self._queue.put_nowait(obs)
            logger.info(
                "Semantic Guard observation queued imdb=%s fp=%s",
                imdb_id,
                fingerprint[:8],
            )
        except asyncio.QueueFull:
            logger.warning("Semantic Guard queue full, dropping observation")
            return

        self._seen_fingerprints.add(fingerprint)

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            try:
                await asyncio.wait_for(
                    self._process_observation(item),
                    timeout=self.timeout_seconds
                )
            except TimeoutError:
                logger.warning("Semantic Guard timeout for imdb=%s", item.imdb_id)
            except Exception as exc:
                logger.error(
                    "Semantic Guard worker error imdb=%s: %s",
                    item.imdb_id,
                    type(exc).__name__,
                )
            finally:
                self._queue.task_done()

    async def _process_observation(self, obs: Observation) -> None:
        if not self._model_loaded:
            await self._load_model()

        try:
            result = await asyncio.to_thread(
                self._analyze,
                obs.target_bytes,
                obs.reference_bytes,
                obs.alass_bytes,
                obs.imdb_id,
                obs.target_id,
            )
            if result:
                await self._write_report(result)
                logger.info(
                    "Semantic Guard analysis completed imdb=%s anchors=%d confidence=%s",
                    obs.imdb_id,
                    result.get("anchor_count", 0),
                    result.get("confidence", False),
                )
        except Exception as exc:
            logger.error(
                "Semantic Guard analysis failed imdb=%s: %s",
                obs.imdb_id,
                type(exc).__name__,
            )

    async def _load_model(self) -> None:
        async with self._lock:
            if self._model_loaded or self._model_load_failed:
                return
            try:
                await asyncio.to_thread(self._load_encoder)
                if self._encoder is not None:
                    self._model_loaded = True
                    logger.info("Semantic Guard model ready")
                else:
                    self._model_load_failed = True
                    logger.warning("Semantic Guard model load returned None")
            except Exception as exc:
                self._model_load_failed = True
                logger.warning("Semantic Guard model load failed: %s", type(exc).__name__)

    def _load_encoder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_id, device="cpu")
        except Exception:
            self._encoder = None

    def _analyze(self, target_bytes: bytes, reference_bytes: bytes, alass_bytes: bytes,
                 imdb_id: str, target_id: str) -> dict[str, Any]:
        if self._encoder is None or self._model_load_failed:
            logger.warning("Semantic Guard analysis skipped: encoder not available")
            return {}

        logger.info(
            "Semantic Guard analysis started imdb=%s",
            imdb_id,
        )

        try:
            from app.services.semantic_guard_core import (
                anchors_from_scores,
                guard_analyze,
                parse_srt,
            )

            arabic = parse_srt(target_bytes)
            reference = parse_srt(reference_bytes)
            alass = parse_srt(alass_bytes)

            logger.info(
                "Semantic Guard cues parsed imdb=%s arabic=%d reference=%d alass=%d",
                imdb_id,
                len(arabic),
                len(reference),
                len(alass),
            )

            encoder = self._encoder
            texts = [c.body for c in arabic + reference]
            vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            scores = vectors[:len(arabic)] @ vectors[len(arabic):].T
            scores[[not c.body for c in arabic], :] = -1
            scores[:, [not c.body for c in reference]] = -1
            anchors = anchors_from_scores(scores)

            report = guard_analyze(arabic, reference, alass, anchors, observe_only=True)

            target_hash = _hash_id(target_id)
            fingerprint = _fingerprint(target_bytes, reference_bytes, alass_bytes, self.model_id)

            result: dict[str, Any] = {
                "imdb_id": imdb_id,
                "target_id_hash": target_hash,
                "fingerprint": fingerprint,
                "anchor_count": len(anchors),
                "confidence": bool(report.get("confident", False)),
                "confidence_reason": str(report.get("reason", "")),
                "slope": report.get("slope"),
                "offset_seconds": report.get("offset_seconds"),
                "inlier_count": report.get("timing_inliers", 0),
                "inlier_ratio": float(report.get("timing_inliers", 0)) / max(len(anchors), 1),
                "coverage": report.get("coverage", 0.0),
                "cv_median_seconds": report.get("semantic_cv_seconds", {}).get("median", 0.0),
                "cv_p90_seconds": report.get("semantic_cv_seconds", {}).get("p90", 0.0),
                "alass_median_seconds": report.get("alass_seconds", {}).get("median", 0.0),
                "alass_p90_seconds": report.get("alass_seconds", {}).get("p90", 0.0),
                "semantic_median_seconds": report.get("original_seconds", {}).get("median", 0.0),
                "disagreement_count": len(report.get("suspicious", [])),
                "proposed_corrections_count": len(report.get("proposed_corrections", [])),
                "proposed_corrections": report.get("proposed_corrections", []),
                "served_result": "alass_unchanged",
                "utc_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "model_identifier": self.model_id,
            }
            return result
        except Exception as exc:
            logger.error(
                "Semantic Guard analysis error imdb=%s: %s",
                imdb_id,
                type(exc).__name__,
            )
            return {}

    async def _write_report(self, report: dict[str, Any]) -> None:
        if not report:
            return

        try:
            await asyncio.to_thread(self._write_report_sync, report)
        except Exception:
            pass

    def _write_report_sync(self, report: dict[str, Any]) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        imdb = _sanitize_filename(report.get("imdb_id", "unknown"))
        target_hash = report.get("target_id_hash", "unknown")
        fingerprint = report.get("fingerprint", "unknown")[:16]
        fname = f"{imdb}_{target_hash}_{fingerprint}.json"
        path = self.report_dir / fname

        allowed = _allowlisted_report(report)
        # Ensure proposed_corrections are numeric only
        if "proposed_corrections" in allowed:
            allowed["proposed_corrections"] = [int(x) for x in allowed["proposed_corrections"] if isinstance(x, int)]
        data = json.dumps(allowed, ensure_ascii=False)
        path.write_text(data, encoding="utf-8")

