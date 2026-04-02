import os

from a2a.types import AgentCard, AgentSkill, AgentCapabilities


INTERVIEW_PREP_AGENT_CARD = AgentCard(
    name="Interview Prep Coach",
    description=(
        "AI-powered interview preparation assistant that analyzes resumes, "
        "recommends study topics, generates practice materials, conducts "
        "mock interviews, and performs job description gap analysis for "
        "software engineering and tech roles."
    ),
    url=os.getenv("AGENT_PUBLIC_URL", "http://localhost:9003"),
    version="1.0.0",
    skills=[
        AgentSkill(
            id="resume-analysis",
            name="Resume Analysis",
            description="Parse and analyze resumes to identify strengths, gaps, and interview focus areas.",
            tags=["resume", "career", "skills", "experience"],
        ),
        AgentSkill(
            id="interview-topics",
            name="Interview Topic Recommendations",
            description="Suggest personalized interview preparation topics based on resume and target role.",
            tags=["interview", "preparation", "topics", "study-plan"],
        ),
        AgentSkill(
            id="study-notes",
            name="Study Notes Generation",
            description="Generate downloadable study materials, practice questions, and concept summaries.",
            tags=["notes", "study", "practice", "preparation"],
        ),
        AgentSkill(
            id="mock-interview",
            name="Mock Interview",
            description="Conduct practice interview sessions with feedback on answers.",
            tags=["mock", "interview", "practice", "feedback"],
        ),
        AgentSkill(
            id="codebase-interview",
            name="Codebase Interview",
            description=(
                "Grill developers on their own codebase. Share a GitHub repo URL and the agent "
                "asks pointed questions about design decisions, edge cases, architecture, and code "
                "quality — one question at a time, like a real technical interviewer."
            ),
            tags=["codebase", "code-review", "github", "technical-interview", "design-decisions"],
        ),
    ],
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
)
