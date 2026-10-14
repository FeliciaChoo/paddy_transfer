"""Small UI helpers shared by every tab."""
import numpy as np
import streamlit as st


def info_panel(what, farmer, manager, limits):
    """The 'what is this telling me' box at the top of each tab. The app is for KPKM / programme administrators only, so only
    the administrator objective is shown (the `farmer` argument is kept so older calls still work and is not displayed)."""
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
    """A click on a map polygon becomes the shared granary.

    Streamlit does not allow changing a widget's value after it has been drawn, so the click is stored in a
    pending slot and applied at the top of app.py on the rerun (that is what updates the sidebar selectbox).
    """
    seen = f"_seen_click_{tag}"
    if not clicked:
        st.session_state[seen] = None
        return
    if clicked != st.session_state.get(seen):
        st.session_state[seen] = clicked
        if clicked != st.session_state.get("granary"):
            st.session_state["_pending_granary"] = clicked
            st.rerun()
