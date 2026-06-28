"""Lightweight contract tests for the vendored helpers in ``app_utils.py``.

Requires the dependencies in ``requirements.txt`` to be installed (importing
``app_utils`` pulls in streamlit/openai/PIL/pydub). Run with: ``pytest``.
"""

import app_utils


def test_work_vars_shape():
    assert app_utils.work_vars["names"]["openai_model"]


def test_work_prompts_has_all_keys():
    prompts = app_utils.work_prompts()
    expected = {
        "intro_summary",
        "date_participants_summary",
        "topic_list_summary",
        "topic_summary",
        "conclusion_summary",
        "summary_begin",
        "summary_end",
        "text_from_audio",
        "text_from_image",
    }
    assert expected.issubset(prompts.keys())


def test_templated_prompts_have_required_placeholders():
    prompts = app_utils.work_prompts()
    # These prompts are consumed via str.format(...) by the app.
    assert "{number_of_topics}" in prompts["topic_list_summary"]
    assert "{topic}" in prompts["topic_summary"]
    assert "{text}" in prompts["summary_begin"] and "{opis}" in prompts["summary_begin"]
    assert "{text}" in prompts["summary_end"] and "{opis_kraj}" in prompts["summary_end"]


def test_plain_prompts_have_no_stray_braces():
    prompts = app_utils.work_prompts()
    for key in ("intro_summary", "conclusion_summary", "text_from_audio", "text_from_image"):
        assert "{" not in prompts[key] and "}" not in prompts[key]
