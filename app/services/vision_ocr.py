from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Iterable

import fitz


LANGUAGE_ALIASES = {
    "eng": "en-US",
    "en": "en-US",
    "en-us": "en-US",
    "vie": "vi-VT",
    "vi": "vi-VT",
    "vi-vn": "vi-VT",
    "vi-vt": "vi-VT",
    "vietnamese": "vi-VT",
}


def is_macos_arm64() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def can_use_vision_ocr() -> bool:
    if not is_macos_arm64():
        return False
    try:
        import Vision  # noqa: F401
        from Foundation import NSData  # noqa: F401
    except Exception:
        return False
    return True


def _split_language_tokens(language: str) -> list[str]:
    tokens = []
    for raw in str(language or "").replace(",", "+").split("+"):
        token = raw.strip().lower()
        if token:
            tokens.append(token)
    return tokens


def format_unsupported_languages(tokens: Iterable[str]) -> str:
    return ", ".join(str(token) for token in tokens)


def _supported_languages() -> set[str]:
    import Vision

    langs, _error = Vision.VNRecognizeTextRequest.supportedRecognitionLanguagesForTextRecognitionLevel_revision_error_(
        Vision.VNRequestTextRecognitionLevelAccurate,
        Vision.VNRecognizeTextRequestRevision3,
        None,
    )
    return {str(lang) for lang in (langs or [])}


def _resolve_languages(language: str) -> tuple[list[str], list[str]]:
    supported = _supported_languages()
    resolved: list[str] = []
    unsupported: list[str] = []

    for token in _split_language_tokens(language):
        candidate = LANGUAGE_ALIASES.get(token, token)
        if candidate in supported:
            if candidate not in resolved:
                resolved.append(candidate)
        else:
            unsupported.append(token)

    if not resolved:
        fallback = "en-US"
        if fallback in supported:
            resolved.append(fallback)
    return resolved, unsupported


@dataclass
class AppleVisionOcr:
    languages: list[str]

    def extract_text_from_fitz_page(self, page: fitz.Page, dpi: int = 200) -> str:
        import Vision
        from Foundation import NSData, NSAutoreleasePool

        pool = NSAutoreleasePool.alloc().init()
        try:
            scale = max(72, int(dpi)) / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            png_bytes = pix.tobytes("png")
            data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))

            observations: list[object] = []
            errors: list[object] = []

            def completion(request: object, error: object) -> None:
                if error is not None:
                    errors.append(error)
                    return
                results = request.results() or []
                observations.extend(list(results))

            request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(completion)
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setRevision_(Vision.VNRecognizeTextRequestRevision3)
            request.setUsesLanguageCorrection_(True)
            if hasattr(request, "setAutomaticallyDetectsLanguage_"):
                request.setAutomaticallyDetectsLanguage_(True)
            if self.languages:
                request.setRecognitionLanguages_(self.languages)

            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
            ok, error = handler.performRequests_error_([request], None)
            if not ok or error is not None:
                raise RuntimeError(error or "Vision OCR request failed")
            if errors:
                raise RuntimeError(errors[0])

            lines: list[tuple[float, float, str]] = []
            for observation in observations:
                candidates = observation.topCandidates_(1)
                if not candidates:
                    continue
                text = str(candidates[0].string()).strip()
                if not text:
                    continue
                box = observation.boundingBox()
                lines.append((float(box.origin.y), float(box.origin.x), text))

            lines.sort(key=lambda item: (-item[0], item[1]))
            return "\n".join(text for _, _, text in lines)
        finally:
            del pool


def build_vision_ocr(language: str = "vie+eng") -> tuple[AppleVisionOcr, list[str]]:
    if not can_use_vision_ocr():
        raise RuntimeError("Apple Vision OCR is only available on macOS arm64 with PyObjC Vision installed")
    languages, unsupported = _resolve_languages(language)
    return AppleVisionOcr(languages=languages), unsupported
