import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from helper import convert_to_html
import re

# Load environment variables from .env file
load_dotenv()


class BlogGenerator:
    def __init__(self, topic: str):
        self.client = OpenAI()
        self.topic = topic
        self.outline = None
        self.blog_content = None
        self.model = "gpt-4o-mini"

        # temp print statements
        print(self.client)
        print(self.model)
        print(self.topic)

    def generate_blog_outline(self):
        system_prompt = """You are an expert content strategist and blog outline creator. Your task is to create a detailed outline for a 1200-word blog post.
        
        Follow these guidelines:
        - Create a clear hierarchical structure using H1, H2, and H3 headings
        - Ensure the outline is comprehensive enough for a 1200-word article
        - Include approximate word counts for each major section
        - Structure should flow logically and maintain reader engagement
        
        Return the outline as a JSON object with the following structure:
        {
            "title": "H1 heading - main title",
            "subtitle": "subtitle",
            "metadata": {"primary_keywords": ["..", ".."], "secondary_keywords": ["..", ".."], "meta_title": "meta title","meta_description": "meta description", "meta_category": "meta category", "meta_tags": ["..", ".."]},
            "sections": [
                {
                    "heading": "H2 heading",
                    "word_count": estimated words,
                    "subsections": [
                        {
                            "heading": "H3 heading",
                            "word_count": estimated words
                        }
                    ]
                }
            ]
        }"""

        try:
            client = OpenAI()
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Create a blog outline for the topic: {self.topic}",
                    },
                ],
                response_format={"type": "json_object"},
                seed=33,
            )

            # Extract and parse the JSON response
            outline = json.loads(completion.choices[0].message.content)
            self.outline = outline
            print(self.outline)
            return outline

        except Exception as e:
            return {"error": str(e)}

    def generate_blog_content(self):
        """
        Generate complete blog content based on the provided outline structure.

        Args:
            outline (dict): Blog outline containing title, subtitle, sections and word counts

        Returns:
            dict: Complete blog content in JSON format
        """

        system_prompt = """You are an expert blog content writer. Your task is to generate complete blog content based on a provided outline structure.

        The input will be a JSON outline like this:
        {
            "title": "Main blog title",
            "subtitle": "Blog subtitle",
            "metadata": "Blog metadata",
            "sections": [
                {
                    "heading": "Section heading",
                    "word_count": target_words,
                    "subsections": [
                        {
                            "heading": "Subsection heading", 
                            "word_count": target_words
                        }
                    ]
                }
            ]
        }

        You must:
        1. Generate high-quality, engaging content for each section and subsection
        2. Follow the specified word count targets closely
        3. Use appropriate transitions between sections
        4. Include relevant examples and actionable tips
        5. Write in a clear, professional tone
        6. Structure the content logically

        Your output must be valid JSON with this structure:
        {
            "title": "title from input",
            "subtitle": "subtitle from input",
            "metadata": "metadata from input",
            "content": [
                {
                    "heading": "section heading",
                    "content": "section content",
                    "subsections": [
                        {
                            "heading": "subsection heading",
                            "content": "subsection content"
                        }
                    ]
                }
            ]
        }"""

        try:
            client = OpenAI()
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Generate complete blog content for this outline: {json.dumps(self.outline)}",
                    },
                ],
                response_format={"type": "json_object"},
                seed=33,
            )

            # Extract and parse the JSON response
            blog_content = json.loads(completion.choices[0].message.content)
            self.blog_content = blog_content
            print(self.blog_content)
            return blog_content

        except Exception as e:
            return {"error": str(e)}

    def create_blog(self):
        outline = self.generate_blog_outline()
        if "error" in outline:
            return outline

        blog_content = self.generate_blog_content()
        if "error" in blog_content:
            return blog_content

        # Generate HTML output
        self.html_output = convert_to_html(self.blog_content)
        return True
