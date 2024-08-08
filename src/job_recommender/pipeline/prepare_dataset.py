from openai import OpenAI
import os
import re
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
topic_prompt = "Give me a list of 5 of a short professional background where each elements only consist one to two sentence with diverse background in with real education, interests, and works experience in Indonesia, but without a name."
client = OpenAI(api_key=OPENAI_API_KEY)

def initialize_topics():
        """Ensure topics are initialized, i.e. topics already exist and are read,
        or a new list of topics is generated.
        """
        topics_path = "topics.txt"
        topic_request_count = 10
        if os.path.exists(topics_path):
            topics = list(
                {
                    line.strip()
                    for line in open(topics_path).readlines()
                    if line.strip()
                }
            )
            
        seen = set([])
        topics = []
        with open(topics_path, "w") as outfile:
            count = topic_request_count
            while count > 0:
                todo = 8 if count >= 8 else count
                responses = [
                        generate_response(topic_prompt)
                        for _ in range(todo)
                    ]
                count -= todo
                for response in responses:
                    if not response:
                        continue
                    for topic in re.findall(
                        r"(?:^|\n)\d+\. (.*?)(?:$|(?=\n\d+\. ))", response, re.DOTALL
                    ):
                        if (
                            not topic
                            or topic.strip().endswith(":")
                            or topic.lower().strip() in seen
                        ):
                            continue
                        seen.add(topic.lower().strip())
                        topics.append(topic)
                        outfile.write(topic.strip() + "\n")

def generate_response(topic_prompt):
        """Call the model endpoint with the specified instruction and return the text response.

        :param instruction: The instruction to respond to.
        :type instruction: str

        :return: Response text.
        :rtype: str
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=1,
            top_p=1,
            messages=[{"role": "user", "content": topic_prompt}]
        )
        
        if (
            not response
            or not response.choices
            or response.choices[0].finish_reason == "length"
        ):
            return None
        text = response.choices[0].message.content

        if text.startswith(("I'm sorry,", "Apologies,", "I can't", "I won't")):
            return None
        return text

initialize_topics()