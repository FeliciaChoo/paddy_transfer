"""One base class for every gate failure, so an interface can catch WorkflowError and show the reason."""


class WorkflowError(Exception):
    """a mandatory step is missing, stale, blocked, or a request is not allowed"""
