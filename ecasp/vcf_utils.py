"""
Helpers for parsing ECASP/OpenSpliceAI VCF INFO fields.
"""

from __future__ import annotations

from typing import Optional, Sequence

PRIMARY_SCORE_INFO_KEY = "ECASP"
LEGACY_SCORE_INFO_KEY = "OpenSpliceAI"
SCORE_INFO_KEYS = (PRIMARY_SCORE_INFO_KEY, LEGACY_SCORE_INFO_KEY)


def extract_info_value(info: str, key: str) -> Optional[str]:
    if not info:
        return None
    tag = f"{key}="
    start = info.find(tag)
    if start == -1:
        return None
    start += len(tag)
    end = info.find(";", start)
    if end == -1:
        end = len(info)
    return info[start:end]


def extract_first_info_value(info: str, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = extract_info_value(info, key)
        if value is not None:
            return value
    return None


def extract_score_info_value(info: str) -> Optional[str]:
    return extract_first_info_value(info, SCORE_INFO_KEYS)
