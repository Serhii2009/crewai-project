from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool, FileWriterTool, FileReadTool
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
load_dotenv()

# 🔑 OpenRouter GPT-4.1 LLM
llm = LLM(
    model="openai/gpt-4.1",
    api_key=os.getenv("CHATGPT_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    custom_llm_provider="openrouter",
    temperature=0.2,
    max_tokens=2048
)

class Content(BaseModel):
    content_type: str = Field(..., description="Type of content (e.g., blog, post, video)")
    topic: str = Field(..., description="Topic of the content")
    target_audience: str = Field(..., description="Target audience")
    tags: List[str] = Field(..., description="Tags to use")
    content: str = Field(..., description="The content itself")

@CrewBase
class TheMarketingCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def _common_tools(self, directory: str = 'resources/drafts'):
        return [
            SerperDevTool(),
            ScrapeWebsiteTool(),
            DirectoryReadTool('resources/drafts'),
            FileWriterTool(),
            FileReadTool()
        ]

    @agent
    def head_of_marketing(self) -> Agent:
        return Agent(
            config=self.agents_config['head_of_marketing'],
            tools=self._common_tools(),
            reasoning=True,
            inject_date=True,
            llm=llm,
            max_rpm=3
        )

    @agent
    def content_creator_social_media(self) -> Agent:
        return Agent(
            config=self.agents_config['content_creator_social_media'],
            tools=self._common_tools(),
            inject_date=True,
            llm=llm,
            max_rpm=3
        )

    @agent
    def content_writer_blogs(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer_blogs'],
            tools=self._common_tools('resources/drafts/blogs'),
            inject_date=True,
            llm=llm,
            max_rpm=3
        )

    @agent
    def seo_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['seo_specialist'],
            tools=self._common_tools(),
            inject_date=True,
            llm=llm,
            max_rpm=3
        )

    @task
    def market_research(self) -> Task:
        return Task(config=self.tasks_config['market_research'], agent=self.head_of_marketing())

    @task
    def prepare_marketing_strategy(self) -> Task:
        return Task(config=self.tasks_config['prepare_marketing_strategy'], agent=self.head_of_marketing())

    @task
    def create_content_calendar(self) -> Task:
        return Task(config=self.tasks_config['create_content_calendar'], agent=self.content_creator_social_media())

    @task
    def prepare_post_drafts(self) -> Task:
        return Task(config=self.tasks_config['prepare_post_drafts'], agent=self.content_creator_social_media(), output_json=Content)

    @task
    def prepare_scripts_for_reels(self) -> Task:
        return Task(config=self.tasks_config['prepare_scripts_for_reels'], agent=self.content_creator_social_media(), output_json=Content)

    @task
    def content_research_for_blogs(self) -> Task:
        return Task(config=self.tasks_config['content_research_for_blogs'], agent=self.content_writer_blogs())

    @task
    def draft_blogs(self) -> Task:
        return Task(config=self.tasks_config['draft_blogs'], agent=self.content_writer_blogs(), output_json=Content)

    @task
    def seo_optimization(self) -> Task:
        return Task(config=self.tasks_config['seo_optimization'], agent=self.seo_specialist(), output_json=Content)

    @crew
    def marketingcrew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=False,
            planning_llm=llm,
            max_rpm=3
        )

if __name__ == "__main__":
    from datetime import datetime

    inputs = {
        "product_name": "AI Powered Excel Automation Tool",
        "target_audience": "Small and Medium Enterprises (SMEs)",
        "product_description": "A tool that automates repetitive tasks in Excel using AI, saving time and reducing errors.",
        "budget": "Rs. 50,000",
        "current_date": datetime.now().strftime("%Y-%m-%d"),
    }
    crew = TheMarketingCrew()
    crew.marketingcrew().kickoff(inputs=inputs)
    print("✅ Marketing crew has been successfully created and run.")
