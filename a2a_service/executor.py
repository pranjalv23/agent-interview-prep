import logging

from agent_sdk.a2a.executor import StreamingAgentExecutor
from agents.agent import run_query, stream_for_a2a

logger = logging.getLogger("agent_interview_prep.a2a_executor")

class InterviewPrepAgentExecutor(StreamingAgentExecutor):
    """A2A executor that streams interview prep agent responses chunk-by-chunk."""
    def __init__(self):
        super().__init__(run_query_fn=run_query, stream_fn=stream_for_a2a)
