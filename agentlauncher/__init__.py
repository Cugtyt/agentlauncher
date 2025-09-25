import asyncio

from .launcher import AgentLauncher

try:
    import uvloop
except ImportError:
    import logging

    logging.warning("uvloop is not installed, using default asyncio event loop.")
    pass
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

__all__ = ["AgentLauncher"]
