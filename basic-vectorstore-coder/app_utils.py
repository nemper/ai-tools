"""Local, self-contained replacement for the helpers this app used to import
from the private ``myfunc.mojafunkcija`` package.

Only the four symbols actually used by ``Koder.py`` are implemented here:
``st_style``, ``positive_login``, ``init_cond_llm`` and ``show_logo``. The goal
is behavioural parity for an archived tool, not a 1:1 reproduction of the
original private code.
"""

from __future__ import annotations

import os

import streamlit as st
import yaml
from yaml.loader import SafeLoader

# streamlit-authenticator is an optional dependency: the login wrapper only runs
# when DEPLOYMENT_ENVIRONMENT == "Streamlit", so importing it lazily keeps the
# app usable locally without the package installed.
try:  # pragma: no cover - exercised only in the Streamlit-cloud deployment path
    import streamlit_authenticator as stauth
except Exception:  # noqa: BLE001 - any import problem just disables login
    stauth = None


# Default chat models offered in the sidebar selector.
DEFAULT_MODELS = ("gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
# Optional company logo; falls back to a text header when unreachable.
LOGO_URL = os.environ.get("APP_LOGO_URL", "")


def st_style() -> None:
    """Hide Streamlit's default main menu, footer and header chrome."""
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_logo() -> None:
    """Render a small logo (or a text fallback) in the sidebar."""
    with st.sidebar:
        if LOGO_URL:
            try:
                st.image(LOGO_URL, use_column_width=True)
                return
            except Exception:  # noqa: BLE001 - never break the app over a logo
                pass
        st.markdown("### 🖥️ Koder")


def init_cond_llm(key: str | None = None) -> tuple[str, float]:
    """Sidebar selectors for the chat model and temperature.

    Returns ``(model_name, temperature)`` so callers can do
    ``model, temp = init_cond_llm()`` exactly as before.
    """
    with st.sidebar:
        model = st.selectbox(
            "Odaberite model",
            DEFAULT_MODELS,
            key=f"model_{key}",
            help="LLM koji se koristi za generisanje odgovora.",
        )
        temperature = st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key=f"temp_{key}",
            help="Niža temperatura = deterministički odgovor.",
        )
    return model, temperature


def positive_login(main, name: str = " "):
    """Minimal streamlit-authenticator login wrapper around ``config.yaml``.

    On success it runs ``main()`` and returns
    ``(name, authentication_status, username)``; on failure it shows an error and
    returns the same triple with a falsy status. Mirrors the original call shape
    ``positive_login(main, " ")``.
    """
    if stauth is None:
        st.error(
            "streamlit-authenticator nije instaliran. Dodajte ga u requirements "
            "ili pokrenite app bez DEPLOYMENT_ENVIRONMENT=Streamlit."
        )
        st.stop()

    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
        config.get("preauthorized"),
    )

    user_name, authentication_status, username = authenticator.login("Login", "main")
    if authentication_status:
        with st.sidebar:
            authenticator.logout("Logout", "main", key="logout_btn")
        main()
    elif authentication_status is False:
        st.error("Pogrešno korisničko ime ili lozinka.")
    else:
        st.warning("Unesite korisničko ime i lozinku.")

    return user_name, authentication_status, username
