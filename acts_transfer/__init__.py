"""acts_transfer: a framework-agnostic transfer-learning workflow (assessment -> harmonisation -> strategy -> evaluation).

Pure Python on pandas DataFrames, so any interface can drive it (Streamlit, Flask routes, a notebook).
The mandatory order is enforced here, in the backend, not in the interface: see workflow.TransferWorkflow.
"""
from .config import Config, Domain

__all__ = ["Config", "Domain", "TransferWorkflow", "WorkflowError"]


def __getattr__(name):                     # lazy, so importing a single module does not need the whole package
    if name in ("TransferWorkflow", "WorkflowError"):
        from . import workflow
        return getattr(workflow, name)
    raise AttributeError(name)
