"""Tests for pure local helper behavior."""

# Requires dependencies in requirements.txt to be installed.

from app_utils import StreamHandler, StreamlitRedirect, open_file


def test_open_file_round_trips_text(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("zdravo\nsvete", encoding="utf-8")

    assert open_file(str(target)) == "zdravo\nsvete"


def test_streamlit_redirect_strips_ansi_and_accumulates_text():
    redirect = StreamlitRedirect()

    redirect.write("pre ")
    redirect.write("\x1b[31mcrveno\x1b[0m")

    assert redirect.get_output() == "pre crveno"


def test_stream_handler_appends_tokens_to_container():
    class FakeContainer:
        def __init__(self):
            self.markdowns = []

        def markdown(self, text):
            self.markdowns.append(text)

    container = FakeContainer()
    handler = StreamHandler(container, initial_text="A")

    handler.on_llm_new_token("B")
    handler.on_llm_new_token("C")

    assert handler.text == "ABC"
    assert container.markdowns == ["AB", "ABC"]
