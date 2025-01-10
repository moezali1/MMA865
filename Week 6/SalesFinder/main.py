import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class SalesAgent:
    def __init__(self, profile):
        """
        Initialize SalesAgent with user profile.

        Args:
            profile (str): The profile of user.
        """
        # Preprocess the profile text
        self.profile = self._preprocess_profile(profile)

    def _preprocess_profile(self, text):
        """
        Preprocess the profile text to make it suitable for API calls.

        Args:
            text (str): Raw profile text from browser

        Returns:
            str: Cleaned and processed text
        """
        # Remove any extra whitespace and newlines
        text = " ".join(text.split())

        # Remove any special characters that might cause issues
        text = text.replace('"', "'")

        # Ensure text is properly encoded
        text = text.encode("utf-8", errors="ignore").decode("utf-8")

        return text

    def generate_email(self):
        system_prompt = """You are an expert career coach and LinkedIn profile optimization specialist. Your role is to craft highly personalized outreach messages to potential clients based on their LinkedIn profiles.

        When analyzing a profile, focus on:
        1. Professional headline and summary optimization
        2. Experience descriptions and achievement metrics
        3. Skills and endorsements strategy
        4. Profile visibility and searchability
        5. Personal branding elements

        Create a compelling, professional message that:
        - Opens with a personalized observation about their specific profile/background
        - Identifies 2-3 key areas where their profile could be strengthened
        - Provides concrete before/after examples of how you would improve those areas
        - Demonstrates the tangible value and ROI of your premium optimization service
        - Maintains a professional yet conversational LinkedIn tone
        - Includes a clear call-to-action

        Example format for improvements:
        Current: "Project Manager leading teams and delivering results"
        Optimized: "Senior Project Manager driving 25% efficiency gains across $2M+ enterprise initiatives through agile transformation"

        Keep the message concise, value-focused, and tailored to the individual's industry and career level. Avoid generic advice or overly sales-focused language.
        """

        try:
            client = OpenAI()
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"User Profile: {self.profile}",
                    },
                ],
                seed=33,
            )

            # Extract and parse the JSON response
            # message = completion.choices[0].message.content
            return completion.choices[0].message.content

        except Exception as e:
            return {"error": str(e)}
