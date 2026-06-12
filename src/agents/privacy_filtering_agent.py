import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.tools.privacy_regex_tool import regex_privacy_tool


load_dotenv()


AGENT_NAME = "Privacy & Filtering Agent"
AGENT_ROLE = "Protect sensitive personal data before ticket analysis."
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class PrivacyAgentOutput(BaseModel):
    masked_text: str = Field(description="Final text after masking sensitive personal data only.")
    is_valid_ticket: bool = Field(description="Whether the message is useful for analysis.")
    message_type: Literal[
        "official_reply",
        "noise_or_praise",
        "question",
        "complaint_or_request",
    ]
    removed_sensitive_items: list[
        Literal[
            "PHONE",
            "EMAIL",
            "NAME",
            "ID",
            "USERNAME",
            "TICKET_NUMBER",
            "ACCOUNT_NUMBER",
        ]
    ]
    explanation: str = Field(description="Short explanation of the decision.")


def run_regex_privacy_filtering(raw_text: str) -> dict:
    tool_output = regex_privacy_tool.invoke(str(raw_text))
    text = tool_output["masked_text"]

    if any(word in text for word in ["نشكر تواصلك", "نسعد بخدمتك", "تم الرد", "خدمة العملاء"]):
        message_type = "official_reply"
    elif len(text) <= 15 and any(word in text for word in ["شكرا", "شكرًا", "ممتاز", "رائع"]):
        message_type = "noise_or_praise"
    elif any(word in text for word in ["؟", "?", "هل", "كيف", "متى", "ليش", "لماذا"]):
        message_type = "question"
    else:
        message_type = "complaint_or_request"

    return {
        "masked_text": text,
        "is_valid_ticket": message_type not in ["official_reply", "noise_or_praise"] and len(text) >= 8,
        "message_type": message_type,
        "removed_sensitive_items": tool_output["removed_sensitive_items"],
        "explanation": "Regex mode only.",
    }


def run_privacy_filtering_agent(raw_text: str) -> dict:
    tool_output = regex_privacy_tool.invoke(str(raw_text))

    if not os.getenv("OPENAI_API_KEY"):
        return run_regex_privacy_filtering(raw_text)

    prompt = f"""
Agent Name:
{AGENT_NAME}

Agent Role:
{AGENT_ROLE}

System:
Zero Ticket is an AI system that transforms customer voice and support tickets into root causes and executive recommendations to reduce repeated contacts.

Tool Used:
regex_privacy_tool

Regex Tool Result:
{json.dumps(tool_output, ensure_ascii=False)}

Your Task:
Review the Regex Tool result and return the final privacy filtering output.

Rules:
1. Keep existing [PHONE] and [EMAIL] masks if correct.
2. Mask sensitive personal data only.
3. Do not mask organization names.
4. Do not mask platform names.
5. Do not mask program names.
6. Do not mask dates.
7. Do not mask years.
8. Do not mask salaries.
9. Do not mask amounts.
10. Do not mask general numbers.
11. Do not mask any number with fewer than 6 digits.
12. Never use [ORG_NAME], [DATE], or [LOCATION].

Allowed masks:
[PHONE], [EMAIL], [NAME], [ID], [USERNAME], [TICKET_NUMBER], [ACCOUNT_NUMBER]

Mask long numbers only when the context clearly shows they are:
national ID, residence ID, ticket number, request number, complaint number, account number, or transaction number.

Do not mask long numbers when they are:
dates, years, salaries, amounts, counts, or durations.

Mask names only when clearly personal, مثل:
- اسمي فلان
- أنا فلان
- معكم فلان
- حسابي باسم فلان

Message Types:
- official_reply
- noise_or_praise
- question
- complaint_or_request

Important:
If message_type is official_reply or noise_or_praise, is_valid_ticket must be false.

Return structured output only.
"""

    try:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
        )

        structured_llm = llm.with_structured_output(PrivacyAgentOutput)
        result = structured_llm.invoke(prompt)

        return result.model_dump()

    except Exception as error:
        print(f"{AGENT_NAME} failed. Fallback to Regex. Error: {error}")
        return run_regex_privacy_filtering(raw_text)