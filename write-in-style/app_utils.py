"""Local replacements for the private myfunc.mojafunkcija helpers.

This archived tool vendors the small helper surface it used from the private
package so the archive remains self-contained.
"""

from __future__ import annotations

import os
import re

import streamlit as st
import yaml
from langchain.callbacks.base import BaseCallbackHandler
from yaml.loader import SafeLoader

try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None

DEFAULT_MODELS = ("gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def st_style() -> None:
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


def open_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def init_cond_llm(key=None):
    with st.sidebar:
        model = st.selectbox("Odaberite model", DEFAULT_MODELS, key=f"model_{key}")
        temperature = st.slider("Temperatura", 0.0, 2.0, 0.0, 0.1, key=f"temp_{key}")
    return model, temperature


def positive_login(main, name: str = " "):
    if stauth is None:
        st.error("streamlit-authenticator nije instaliran.")
        st.stop()
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config["credentials"], config["cookie"]["name"], config["cookie"]["key"],
        config["cookie"]["expiry_days"], config.get("preauthorized"),
    )
    user_name, authentication_status, username = authenticator.login("Login", "main")
    if authentication_status:
        with st.sidebar:
            authenticator.logout("Logout", "main", key="logout_btn")
        main()
    elif authentication_status is False:
        st.error("Pogresno korisnicko ime ili lozinka.")
    else:
        st.warning("Unesite korisnicko ime i lozinku.")
    return user_name, authentication_status, username


class StreamHandler(BaseCallbackHandler):
    """Stream LLM tokens into a Streamlit container as they arrive."""

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class StreamlitRedirect:
    """File-like sink that strips ANSI codes so verbose agent stdout can be shown in Streamlit."""

    def __init__(self):
        self.output = ""

    def write(self, text: str) -> None:
        self.output += re.sub(r"\x1b\[[0-9;]*m", "", text)

    def flush(self) -> None:
        pass

    def get_output(self) -> str:
        return self.output
