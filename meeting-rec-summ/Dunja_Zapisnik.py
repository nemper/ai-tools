"""Reference module for the "asistenti" helper functions.

Historically this file held standalone reference copies of the helpers that the
private ``myfunc.asistenti`` package provided (audio transcription, image
description, etc.). Those implementations now live, canonically and
self-contained, in ``app_utils.py`` — there is a single source of truth and no
external private dependency.

This module is kept for backwards compatibility / documentation: it simply
re-exports the canonical helpers so any code doing ``from Dunja_Zapisnik import
priprema`` keeps working.
"""

from app_utils import (  # noqa: F401 - re-exported for backwards compatibility
    delete_mp3_files,
    generate_corrected_transcript,
    mprompts,
    priprema,
    read_local_image,
    read_url_image,
    transkript,
    work_prompts,
    work_vars,
)

__all__ = [
    "priprema",
    "transkript",
    "read_local_image",
    "read_url_image",
    "generate_corrected_transcript",
    "delete_mp3_files",
    "work_prompts",
    "work_vars",
    "mprompts",
]
