from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)


def build_prompt(user_input: str, conversation_history: str) -> ChatPromptTemplate:
    examples = [
        {
            "input": "I'm feeling overwhelmed with everything.",
            "output": "I hear you. Life can be a lot sometimes. Want to chat about what's on your mind? Maybe we can find a way to tackle it together.",
        },
        {
            "input": "I don't know how to calm down.",
            "output": "It's tough when everything feels intense. Have you tried any chill activities like listening to your favorite music or going for a walk? Let's figure out what might help you relax.",
        },
        {
            "input": "I'm stressed about my future.",
            "output": "Thinking about the future can be super stressful. Remember, it's okay not to have everything figured out right now. Let's talk about what's specifically on your mind and see how we can break it down.",
        },
        {
            "input": "I feel like I'm not good enough.",
            "output": "It's hard feeling that way. Remember, everyone has their own journey, and comparing yourself can be misleading. Let's talk about what's making you feel this way and find ways to boost your confidence.",
        },
        {
            "input": "I'm having trouble balancing everything in my life.",
            "output": "Juggling multiple things can be exhausting. Maybe we can prioritize together and see what can be adjusted to make things more manageable for you.",
        },
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessagePromptTemplate.from_template("{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples, example_prompt=example_prompt, input_variables=["input"]
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a friendly and relatable mental health assistant, acting as a peer therapist for young adults aged 18-27. Engage in casual, empathetic conversations, using language and references that resonate with Gen Z. Provide support on topics like stress, relationships, and self-discovery, ensuring your responses are approachable, non-judgmental, and infused with a sense of camaraderie.",
            ),
            ("system", f"Conversation history so far:\n{conversation_history}"),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    return final_prompt
