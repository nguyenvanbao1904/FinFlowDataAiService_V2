from __future__ import annotations

import platform
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

try:
    import objc  # type: ignore
except Exception:  # pragma: no cover
    objc = None  # type: ignore[assignment]


class VisionOcrUnavailableError(RuntimeError):
    """Raised when Apple Vision OCR cannot be used in current runtime."""


_TESSERACT_TO_VISION_LANG = {
    "vie": "vi-VN",
    "eng": "en-US",
    "fra": "fr-FR",
    "deu": "de-DE",
    "spa": "es-ES",
    "ita": "it-IT",
    "por": "pt-BR",
    "jpn": "ja-JP",
    "kor": "ko-KR",
    "chi_sim": "zh-Hans",
    "chi_tra": "zh-Hant",
}


def _autorelease_pool():
    if objc is None:
        return nullcontext()
    try:
        return objc.autorelease_pool()  # type: ignore[no-any-return]
    except Exception:
        return nullcontext()


def is_macos_arm64() -> bool:
    machine = platform.machine().lower()
    return platform.system() == "Darwin" and machine in {"arm64", "aarch64"}


def can_use_vision_ocr() -> bool:
    if not is_macos_arm64():
        return False
    try:
        __import__("Vision")
        __import__("Foundation")
    except Exception:
        return False
    return True


def parse_vision_languages(language_spec: str) -> tuple[list[str], list[str]]:
    raw_tokens = [token.strip() for token in re.split(r"[+,; ]+", str(language_spec or "")) if token.strip()]
    if not raw_tokens:
        return ["vi-VN", "en-US"], []

    resolved: list[str] = []
    unsupported: list[str] = []
    for token in raw_tokens:
        key = token.lower()
        mapped = _TESSERACT_TO_VISION_LANG.get(key)
        if mapped:
            if mapped not in resolved:
                resolved.append(mapped)
            continue

        # Accept BCP-47 style input directly, e.g. vi-VN, en-US.
        if re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?", token):
            normalized = token.replace("_", "-")
            if normalized not in resolved:
                resolved.append(normalized)
            continue

        unsupported.append(token)

    if not resolved:
        resolved = ["vi-VN", "en-US"]
    return resolved, unsupported


@dataclass
class AppleVisionOcr:
    languages: list[str]
    recognition_level: str = "accurate"
    uses_language_correction: bool = True

    def __post_init__(self) -> None:
        if not can_use_vision_ocr():
            raise VisionOcrUnavailableError(
                "Apple Vision OCR requires macOS ARM64 with pyobjc Vision frameworks installed."
            )

        from Foundation import NSData  # type: ignore
        from Vision import VNImageRequestHandler, VNRecognizeTextRequest  # type: ignore

        self._NSData = NSData
        self._VNImageRequestHandler = VNImageRequestHandler
        self._VNRecognizeTextRequest = VNRecognizeTextRequest
        self._CGImageSourceCreateWithData = None
        self._CGImageSourceCreateImageAtIndex = None
        try:
            from Quartz import CGImageSourceCreateImageAtIndex, CGImageSourceCreateWithData  # type: ignore

            self._CGImageSourceCreateWithData = CGImageSourceCreateWithData
            self._CGImageSourceCreateImageAtIndex = CGImageSourceCreateImageAtIndex
        except Exception:
            # Fallback to VNImageRequestHandler initWithData if Quartz bridge is unavailable.
            pass

        try:
            from Vision import (  # type: ignore
                VNRequestTextRecognitionLevelAccurate,
                VNRequestTextRecognitionLevelFast,
            )
        except Exception:
            # Compatibility fallback across pyobjc versions.
            VNRequestTextRecognitionLevelAccurate = 1
            VNRequestTextRecognitionLevelFast = 0

        self._level_accurate = VNRequestTextRecognitionLevelAccurate
        self._level_fast = VNRequestTextRecognitionLevelFast

    def _build_request(self, *, use_languages: bool = True):  # type: ignore[no-untyped-def]
        request = self._VNRecognizeTextRequest.alloc().init()
        level = self._level_accurate if self.recognition_level != "fast" else self._level_fast
        try:
            request.setRecognitionLevel_(level)
        except Exception:
            pass
        try:
            request.setUsesLanguageCorrection_(bool(self.uses_language_correction))
        except Exception:
            pass
        if use_languages and self.languages:
            try:
                request.setRecognitionLanguages_(self.languages)
            except Exception:
                pass
        return request

    def _build_handler(self, png_bytes: bytes):
        data = self._NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
        if self._CGImageSourceCreateWithData is not None and self._CGImageSourceCreateImageAtIndex is not None:
            src = self._CGImageSourceCreateWithData(data, None)
            if src is not None:
                cg_image = self._CGImageSourceCreateImageAtIndex(src, 0, None)
                if cg_image is not None:
                    return self._VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        return self._VNImageRequestHandler.alloc().initWithData_options_(data, None)

    def _perform_request(self, png_bytes: bytes, *, use_languages: bool) -> tuple[object | None, object | None]:
        request = self._build_request(use_languages=use_languages)
        handler = self._build_handler(png_bytes)
        ok, error = handler.performRequests_error_([request], None)
        if ok:
            return request, None

        # Retry with explicit empty options for older pyobjc bindings.
        data = self._NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
        handler = self._VNImageRequestHandler.alloc().initWithData_options_(data, {})
        ok, error = handler.performRequests_error_([request], None)
        if ok:
            return request, None
        return None, error

    def extract_text_from_png_bytes(self, png_bytes: bytes) -> str:
        if not png_bytes:
            return ""

        with _autorelease_pool():
            request, error = self._perform_request(png_bytes, use_languages=True)
            if request is None and self.languages:
                request, error = self._perform_request(png_bytes, use_languages=False)
            if request is None:
                raise RuntimeError(f"Vision OCR request failed: {error}")

            observations = list(request.results() or [])
            lines: list[tuple[float, float, str]] = []
            for obs in observations:
                candidates = list(obs.topCandidates_(1) or [])
                if not candidates:
                    continue
                text = str(candidates[0].string() or "").strip()
                if not text:
                    continue
                box = obs.boundingBox()
                # Convert to top-left sorting coordinates.
                y_top = 1.0 - float(box.origin.y + box.size.height)
                x_left = float(box.origin.x)
                lines.append((y_top, x_left, text))

            lines.sort(key=lambda item: (round(item[0], 4), item[1]))
            return "\n".join(item[2] for item in lines).strip()

    def extract_text_from_fitz_page(
        self,
        page: object,
        *,
        dpi: int = 200,
    ) -> str:
        # `page` is expected to be a PyMuPDF Page object.
        import fitz

        scale = max(72, int(dpi)) / 72.0
        with _autorelease_pool():
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            png_bytes = pix.tobytes("png")
            del pix
        return self.extract_text_from_png_bytes(png_bytes)


def build_vision_ocr(language_spec: str, *, recognition_level: str = "accurate") -> tuple[AppleVisionOcr, list[str]]:
    languages, unsupported = parse_vision_languages(language_spec)
    return AppleVisionOcr(languages=languages, recognition_level=recognition_level), unsupported


def format_unsupported_languages(tokens: Iterable[str]) -> str:
    values = [str(token).strip() for token in tokens if str(token).strip()]
    return ", ".join(values)
