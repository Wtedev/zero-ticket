import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


AGENT_NAME = "Classification Agent"
AGENT_ROLE = "Classify customer issues into journey stage, solution type, priority, and responsible owner team."
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class ClassificationAgentOutput(BaseModel):
    problem_category: str = Field(description="Main issue category.")
    journey_stage: Literal[
        "awareness",
        "registration",
        "login_access",
        "application_submission",
        "eligibility_review",
        "service_usage",
        "completion",
        "support_follow_up",
        "unknown",
    ]
    solution_type: list[
        Literal[
            "تحسين تجربة وواجهة المستخدم",
            "تبسيط الإجراءات",
            "أتمتة الإشعارات",
            "تحسين رسائل الخطأ أو الرفض",
            "محتوى إرشادي داخل الخدمة / Knowledge Base",
        ]
    ]
    priority: Literal["low", "medium", "high"]
    owner_team: Literal[
        "product",
        "ux_ui",
        "operations",
        "customer_support",
        "technical",
        "content",
        "unknown",
    ]
    confidence_score: float = Field(ge=0, le=1)


def run_rule_based_classification(masked_text: str) -> dict:
    text = str(masked_text)

    output = {
        "problem_category": "general_service_confusion",
        "journey_stage": "unknown",
        "solution_type": ["محتوى إرشادي داخل الخدمة / Knowledge Base"],
        "priority": "medium",
        "owner_team": "product",
        "confidence_score": 0.55,
    }

    if any(word in text for word in ["شهادة", "الشهادة", "أنهيت", "انهيت", "إكمال", "اكمال", "الدورة"]):
        output.update({
            "problem_category": "certificate_or_completion_status_unclear",
            "journey_stage": "completion",
            "solution_type": ["أتمتة الإشعارات", "تحسين تجربة وواجهة المستخدم"],
            "priority": "high",
            "owner_team": "product",
        })

    elif any(word in text for word in ["دخول", "تسجيل", "حساب", "رمز", "كلمة المرور", "OTP"]):
        output.update({
            "problem_category": "login_or_account_access_issue",
            "journey_stage": "login_access",
            "solution_type": ["تحسين رسائل الخطأ أو الرفض", "تحسين تجربة وواجهة المستخدم"],
            "priority": "high",
            "owner_team": "technical",
        })

    elif any(word in text for word in ["رقم الطلب", "حالة الطلب", "تذكرة", "لم يتم تحديث", "تحديث الحالة"]):
        output.update({
            "problem_category": "request_status_not_clear",
            "journey_stage": "support_follow_up",
            "solution_type": ["أتمتة الإشعارات", "تحسين تجربة وواجهة المستخدم"],
            "priority": "high",
            "owner_team": "operations",
        })

    elif any(word in text for word in ["رفض", "غير مؤهل", "استبعاد", "سبب الرفض"]):
        output.update({
            "problem_category": "rejection_reason_unclear",
            "journey_stage": "eligibility_review",
            "solution_type": ["تحسين رسائل الخطأ أو الرفض", "تبسيط الإجراءات"],
            "priority": "high",
            "owner_team": "operations",
        })

    return output


def run_classification_agent(masked_text: str) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        return run_rule_based_classification(masked_text)

    prompt = f"""
Agent Name:
{AGENT_NAME}

Agent Role:
{AGENT_ROLE}

System:
Zero Ticket is an AI system that transforms customer voice and support tickets into root causes and executive recommendations to reduce repeated contacts.

Core Principle:
Zero Ticket does not suggest replies to tickets.
Zero Ticket suggests operational, product, UX, automation, content, or process solutions that reduce the reason customers contact support in the future.

Your Task:
Classify the masked customer message.

Do NOT:
- Do not write a response to the customer.
- Do not suggest "contact support" as a solution.
- Do not produce FAQ answers.
- Do not solve the individual ticket.
- Do not focus on replying.

Do:
- Identify the problem category.
- Identify the beneficiary journey stage.
- Select one or more solution types.
- Select the owner team responsible for fixing the root cause.
- Set priority based on impact and recurrence potential.

Allowed solution_type values:
- تحسين تجربة وواجهة المستخدم
- تبسيط الإجراءات
- أتمتة الإشعارات
- تحسين رسائل الخطأ أو الرفض
- محتوى إرشادي داخل الخدمة / Knowledge Base

Solution Type Meaning:
- أتمتة الإشعارات: when the issue can be reduced by automatic status updates, email/SMS/app notifications, or proactive alerts.
- تحسين تجربة وواجهة المستخدم: when users need clearer status, buttons, pages, dashboards, progress indicators, or next steps inside the product.
- تبسيط الإجراءات: when the process itself is long, confusing, repetitive, or has unnecessary steps.
- تحسين رسائل الخطأ أو الرفض: when the user does not understand why something failed, was rejected, or what to do next.
- محتوى إرشادي داخل الخدمة / Knowledge Base: when users need embedded guidance, help center content, tooltips, or self-service explanations. This is not a ticket reply.

Examples:
Input: "خلصت الدورة وما وصلتني الشهادة"
Output direction:
problem_category = certificate_or_completion_status_unclear
journey_stage = completion
solution_type = ["أتمتة الإشعارات", "تحسين تجربة وواجهة المستخدم"]
owner_team = product

Input: "رقم الطلب له أسبوع ما تغيرت حالته"
Output direction:
problem_category = request_status_not_clear
journey_stage = support_follow_up
solution_type = ["أتمتة الإشعارات", "تحسين تجربة وواجهة المستخدم"]
owner_team = operations

Input: "تم رفض طلبي وما أعرف السبب"
Output direction:
problem_category = rejection_reason_unclear
journey_stage = eligibility_review
solution_type = ["تحسين رسائل الخطأ أو الرفض", "تبسيط الإجراءات"]
owner_team = operations

Masked text:
{masked_text}

Return structured output only.
"""

    try:
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
        )

        structured_llm = llm.with_structured_output(ClassificationAgentOutput)
        result = structured_llm.invoke(prompt)

        return result.model_dump()

    except Exception as error:
        print(f"{AGENT_NAME} failed. Fallback to rules. Error: {error}")
        return run_rule_based_classification(masked_text)