import logging

from agent_sdk.a2a.factory import create_a2a_app as _create

from .agent_card import INTERVIEW_PREP_AGENT_CARD
from .executor import InterviewPrepAgentExecutor

logger = logging.getLogger("agent_interview_prep.a2a_server")


def create_a2a_app():
    """Build the A2A Starlette application for the interview prep agent."""
    app = _create(INTERVIEW_PREP_AGENT_CARD, InterviewPrepAgentExecutor, "agent_interview_prep")
    logger.info("A2A application created for Interview Prep Agent")
    return app
