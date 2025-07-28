import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool

# 🧠 AI model setup (via OpenRouter)
llm = LLM(
    model="openai/gpt-4.1",
    api_key=os.getenv("CHATGPT_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    custom_llm_provider="openrouter",
    temperature=0.3,
    max_tokens=1200
)

@CrewBase
class InsightCrew():
    """AI Trends Insight Generator Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            tools=[SerperDevTool()],
            llm=llm,
            verbose=True
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["summarizer"],
            llm=llm,
            verbose=True
        )

    @task
    def fetch_insights(self) -> Task:
        return Task(
            config=self.tasks_config["fetch_insights"],
            agent=self.analyst()
        )

    @task
    def summarize_insights(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_insights"],
            agent=self.summarizer(),
            context=[self.fetch_insights()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.analyst(), self.summarizer()],
            tasks=[self.fetch_insights(), self.summarize_insights()],
            process=Process.sequential,
            verbose=True
        )

if __name__ == "__main__":
    insight_crew = InsightCrew()
    insight_crew.crew().kickoff(inputs={"topic": "AI insights for 2025"})
