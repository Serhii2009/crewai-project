import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

# 🔑 OpenRouter GPT-4.1 LLM
llm = LLM(
    model="openai/gpt-4.1",
    api_key=os.getenv("CHATGPT_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    custom_llm_provider="openrouter",
    temperature=0.2,
    max_tokens=1000
)

@CrewBase
class BlogCrew():
    """Blog writing crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],
            tools=[SerperDevTool()],
            llm=llm,
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer_agent"],
            llm=llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher()
        )

    @task
    def blog_task(self) -> Task:
        return Task(
            config=self.tasks_config["blog_task"],
            agent=self.writer(),
            context=[self.research_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher(), self.writer()],
            tasks=[self.research_task(), self.blog_task()],
            process=Process.sequential,
            verbose=True
        )

if __name__ == "__main__":
    blog_crew = BlogCrew()
    blog_crew.crew().kickoff(inputs={"topic": "The future of electrical vehicles"})
