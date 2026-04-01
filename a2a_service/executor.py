import logging
import traceback

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)

from agents.agent import run_query

logger = logging.getLogger("agent_interview_prep.a2a_executor")


class InterviewPrepExecutor(AgentExecutor):
    """A2A executor that bridges incoming A2A tasks to the interview prep agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("A2A execute — task_id='%s'", context.task_id)

        query = context.get_user_input()
        if not query:
            logger.error("No text content found in the request")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(state=TaskState.failed),
                )
            )
            return

        task_metadata = getattr(context, "task", {}).get("metadata", {}) if hasattr(context, "task") else {}
        user_id = task_metadata.get("user_id") or context.context_id or context.task_id
        mode = task_metadata.get("mode") or "coach"

        logger.info("A2A execute — task_id='%s', user_id='%s', mode='%s'", 
                    context.task_id, user_id, mode)

        try:
            result = await run_query(query, session_id=session_id, user_id=user_id)
            # Interview prep agent might not use mode yet, but we pass it anyway if added later
            # result = await run_query(query, session_id=session_id, user_id=user_id, mode=mode)
            response_text = result["response"]

            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    artifact=Artifact(
                        parts=[Part(root=TextPart(text=response_text))],
                    ),
                    last_chunk=True,
                )
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(state=TaskState.completed),
                )
            )
        except Exception as e:
            logger.error("A2A execution failed: %s\n%s", e, traceback.format_exc())
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(state=TaskState.failed),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError(message="Cancel is not supported.")
