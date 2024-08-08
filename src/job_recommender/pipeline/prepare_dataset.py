from groq import Groq

client = Groq()

chat_completion = client.chat.completions.create(
    #
    # Required parameters
    #
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": """Generate a CV for applying based on the given description.
Create resume with real university and real company in Indonesia.
The CV should be contains only of sections: Education, Experience, Projects, and Skills.
Each section should be separated by a new line and capitalized."""
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": """Fresh graduate mathematics in Indonesia.
He want to get a job as AI engineer and Data engineer in big tech company.
He had internship experience in startup as backend engineer.""",
        }
    ],

    # The language model which will generate the completion.
    model="llama-3.1-70b-versatile",

    #
    # Optional parameters
    #

    # Controls randomness: lowering results in less random completions.
    # As the temperature approaches zero, the model will become deterministic
    # and repetitive.
    temperature=1,

    # The maximum number of tokens to generate. Requests can use up to
    # 32,768 tokens shared between prompt and completion.
    max_tokens=1024,

    # Controls diversity via nucleus sampling: 0.5 means half of all
    # likelihood-weighted options are considered.
    top_p=1,

    # A stop sequence is a predefined or user-specified text string that
    # signals an AI to stop generating content, ensuring its responses
    # remain focused and concise. Examples include punctuation marks and
    # markers like "[end]".
    stop=None,

    # If set, partial message deltas will be sent.
    stream=False,
)

print(chat_completion.choices[0].message.content)

class GenerateSyntheticResume:
    def __init__(self):
        self.instruct = """Generate a CV for applying based on the given description.
Create resume with real university and real company in Indonesia.
The CV should be contains only of sections: Education, Experience, Projects, and Skills.
Each section should be separated by a new line and capitalized."""

    def generate_description(self):
        pass

    def generate_resume(self):
        pass

class GenerateInstructionLabels:
    def __init__(self):
        pass

    def generate_feedback(self):
        pass