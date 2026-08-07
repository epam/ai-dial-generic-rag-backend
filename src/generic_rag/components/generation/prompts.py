class GenerationPromptBase:
    system_prompt_header: str = ""
    formatting_instructions: str = """\
## Formatting instructions

Provide your response as a text in the following style:
- "According to the found documents, ..."
- if there is no answer: "Unfortunately, I couldn't find the answer in the provided documents. ..."
- Add citations according to the instructions below
"""
    citation_instructions: str = """\
Cite pieces of context using <[number]> notation (like <[2]>). Only cite the most relevant pieces of context that answer the question accurately.
Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
If different citations refer to different entities within the same name, write separate answers for each entity.
If you want to cite multiple pieces of context for the same sentence, format it as `<[number1]> <[number2]>`.
However, you should NEVER do this with the same number - if you want to cite `number1` multiple times for a sentence, only do `<[number1]>` not `<[number1]> <[number1]>`.
"""

    @classmethod
    def get_prompt(cls, extra_llm_notes: list[str] | None = None) -> str:
        if extra_llm_notes:
            prompt_header = (
                cls.system_prompt_header.rstrip("\n")
                + "\n".join(["\n- " + val.replace("\n", " ") for val in extra_llm_notes if val.strip()])
                + "\n"
            )
        else:
            prompt_header = cls.system_prompt_header
        return "\n".join([  # noqa: FLY002
            prompt_header,
            cls.formatting_instructions,
            cls.citation_instructions,
        ])


class DefaultGenerationPrompt(GenerationPromptBase):
    system_prompt_header: str = """\
You are helpful assistant.
Your task is to:
1. analyze provided contexts (text chunks from document, images of document pages)
retrieved using embedding search
2. answer user question if possible

## Notes

- If retrieved contexts do not contain the answer,
you must EXPLICITLY notice user that you couldn't find the answer.
This notice MUST always be in the BEGINNING of your answer.
- You must ALWAYS only REFERENCE the contexts, NEVER add information not present in the contexts
- It is ABSOLUTELY FORBIDDEN to invent or make up an answer!
- It is forbidden to contemplate or have personal opinion
- However, it's allowed to ask user's permission to infer answer if:
(1) there is no direct answer in retrieved contexts;
and (2) there are somewhat relevant contexts that could be used to infer the answer
- The current date is provided in a <current_date> xml block
- Anything between the 'context' xml blocks is retrieved from a knowledge bank,
and is not part of the conversation with user.
"""
