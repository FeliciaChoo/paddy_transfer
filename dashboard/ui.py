"""Small UI helpers shared by every tab."""
import numpy as np
import streamlit as st


def inject_gemini_sidebar_css():
    """Clean, well-spaced, flush left-aligned sidebar styling without overlapping labels."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            background-color: #0B2117 !important;
        }
        /* Comfortable breathing room between sidebar blocks */
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.65rem !important;
        }
        /* Style native selectbox labels cleanly instead of overlapping markdown headers */
        section[data-testid="stSidebar"] [data-testid="stSelectbox"] label p {
            color: #94A89E !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            margin-bottom: 4px !important;
        }
        /* Flush left-aligned navigation buttons */
        section[data-testid="stSidebar"] div.stButton > button {
            width: 100% !important;
            background-color: transparent !important;
            color: #CBD5E1 !important;
            border: 1px solid transparent !important;
            border-radius: 8px !important;
            padding: 0.5rem 0.75rem !important;
            min-height: 38px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            box-shadow: none !important;
            margin: 0 !important;
            transition: background-color 0.14s ease, color 0.14s ease;
        }
        section[data-testid="stSidebar"] div.stButton > button > div {
            width: 100% !important;
            display: flex !important;
            justify-content: flex-start !important;
        }
        section[data-testid="stSidebar"] div.stButton > button p {
            width: 100% !important;
            text-align: left !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            margin: 0 !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: rgba(255, 255, 255, 0.07) !important;
            color: #F8FAFC !important;
        }
        /* Active item pill highlight */
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background-color: rgba(159, 206, 52, 0.18) !important;
            border: 1px solid rgba(159, 206, 52, 0.45) !important;
            color: #D9F99D !important;
        }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] p {
            color: #D9F99D !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_dark_sidebar_pill_css = inject_gemini_sidebar_css


def render_gemini_nav_item(label: str, key: str, active: bool) -> bool:
    """Render a clean, single-line left-aligned sidebar navigation item."""
    return st.button(
        label,
        key=key,
        type="primary" if active else "secondary",
        width="stretch",
    )


def info_panel(what, farmer, manager, limits):
    """The 'what is this telling me' box at the top of each tab."""
    with st.expander("What is this telling me, and what should I do with it?", expanded=False):
        st.markdown(f"**What this shows.** {what}")
        st.markdown(f"**What KPKM / programme administrators can do with it**\n\n{manager}")
        st.caption(f"How far to trust it: {limits}")


def pct(x, digits=0):
    """'12%' for a fraction, 'n/a' when it is missing."""
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}%}"


def selected_granary():
    """The granary chosen in the sidebar (or by clicking a map). One value shared by all tabs."""
    return st.session_state["granary"]


def apply_map_click(clicked, tag):
    """A click on a map polygon becomes the shared granary."""
    seen = f"_seen_click_{tag}"
    if not clicked:
        st.session_state[seen] = None
        return
    if clicked != st.session_state.get(seen):
        st.session_state[seen] = clicked
        if clicked != st.session_state.get("granary"):
            st.session_state["_pending_granary"] = clicked
            st.rerun()