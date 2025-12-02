import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =============================================================================
# Function: build_chain
# Purpose: Creates the complete AI pipeline:
#   ChatPromptTemplate → Model → StrOutputParser
# This returns a single object (a "chain") you can invoke with .invoke().
# =============================================================================
def build_chain():

    # ------------------------------------------------------------
    # ChatOpenAI:
    #   - Uses GPT-4o-mini via GitHub Models
    #   - base_url points to GitHub's OpenAI-compatible endpoint
    #   - temperature controls creativity (0 = factual, 1 = spicy)
    # ------------------------------------------------------------
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["GITHUB_TOKEN"],  # pulled from environment
        base_url="https://models.inference.ai.azure.com",
        temperature=0.8,  # playful tone for roasts
    )

    # ------------------------------------------------------------
    # ChatPromptTemplate:
    #   - Defines the structure of the conversation
    #   - ("system", ...)   → sets behavior/style of the AI
    #   - ("human", ...)    → user-facing message with variables
    #   - Variables inside {} get replaced at runtime
    # ------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a witty but kind social media assistant. "
            "Write playful, safe, non-offensive roast tweets. "
            "Keep responses under 240 characters."
        ),
        (
            "human",
            "Roast {target} about {topic} in a {mood} tone."
        ),
    ])

    # ------------------------------------------------------------
    # StrOutputParser:
    #   - Converts the model’s AIMessage into a clean string
    #   - Without this, you'd get a full AIMessage object
    # ------------------------------------------------------------
    parser = StrOutputParser()

    # ------------------------------------------------------------
    # LCEL pipeline:
    #   prompt → model → parser
    #   The "|" operator connects components into a single chain.
    #
    #   When you call chain.invoke({...}), LangChain:
    #   1. Fills the prompt with your variables
    #   2. Sends it to the model
    #   3. Parses the output into a final string
    # ------------------------------------------------------------
    return prompt | model | parser


# =============================================================================
# Main runtime function
# =============================================================================
def main():

    # Build the reusable roast chain
    chain = build_chain()

    # Ask the user for all variables needed by the prompt
    target = input("Who should be roasted (@handle or name)? ")
    topic = input("What topic are they obsessed with? ")
    mood = input("Mood (playful / spicy / gentle)? ") or "playful"

    # ------------------------------------------------------------
    # chain.invoke():
    #   - Fills in the variables in the prompt
    #   - Sends the completed prompt → model
    #   - Returns the final string (thanks to the parser)
    # ------------------------------------------------------------
    roast = chain.invoke({
        "target": target,
        "topic": topic,
        "mood": mood,
    })

    # Display the final result
    print("\n--- Generated Roast Tweet ---")
    print(roast)


# Run main() only when executed directly
if __name__ == "__main__":
    main()
