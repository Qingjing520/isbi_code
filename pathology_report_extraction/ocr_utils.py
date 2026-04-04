from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from pdf_utils import render_page


def deskew_binary_image(binary: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary < 250))
    if coords.size == 0:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.2 or abs(angle) > 12:
        return binary

    (height, width) = binary.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(binary, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_for_ocr(image_bgr: np.ndarray, deskew: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
    normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        13,
    )
    if deskew:
        binary = deskew_binary_image(binary)
    return binary


@lru_cache(maxsize=1)
def get_ocr_engine() -> RapidOCR:
    return RapidOCR()


def ocr_image(image: np.ndarray) -> list[dict[str, Any]]:
    result, _ = get_ocr_engine()(image)
    lines: list[dict[str, Any]] = []
    for item in result or []:
        box = item[0]
        text = item[1].strip()
        score = float(item[2]) if len(item) > 2 else None
        if not text:
            continue
        lines.append(
            {
                'text': text,
                'score': score,
                'bbox': [[float(point[0]), float(point[1])] for point in box],
            }
        )
    return lines


def join_ocr_lines(lines: list[dict[str, Any]]) -> str:
    return '\n'.join(line['text'] for line in lines if line['text'].strip()).strip()


def ocr_page(page: fitz.Page, zoom: float = 2.5, deskew: bool = True) -> dict[str, Any]:
    image_bgr = render_page(page, zoom=zoom)
    processed = preprocess_for_ocr(image_bgr, deskew=deskew)
    lines = ocr_image(processed)
    return {
        'lines': lines,
        'text': join_ocr_lines(lines),
        'image_shape': list(processed.shape),
    }
