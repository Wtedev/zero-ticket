import streamlit as st
import pandas as pd

from src.agents.privacy_filtering_agent import (
    run_privacy_filtering_agent,
    run_regex_privacy_filtering,
)


st.set_page_config(
    page_title="Zero Ticket AI Agent",
    page_icon="🎫",
    layout="wide",
)


st.title("🎫 Zero Ticket AI Agent")

st.caption(
    "Zero Ticket لا يصنف التذاكر فقط، بل يحول صوت المستفيد من تذاكر متفرقة "
    "إلى أسباب جذرية وتوصيات تنفيذية تقلل تكرار التواصل مستقبلًا."
)

st.info(
    "في بيئة الهاكاثون، ولحماية البيانات، استخدمنا بيانات عامة ومحاكاة واقعية "
    "لإثبات قابلية الحل. وفي التطبيق الفعلي يمكن ربط Zero Ticket بأنظمة التذاكر الرسمية."
)


uploaded_file = st.file_uploader(
    "ارفعي ملف التذاكر أو صوت المستفيد بصيغة Excel أو CSV",
    type=["xlsx", "csv"],
)


def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    return None


if uploaded_file is None:
    st.warning("ارفعي ملف Excel أو CSV للبدء.")
    st.stop()


df = load_data(uploaded_file)

if df is None:
    st.error("صيغة الملف غير مدعومة.")
    st.stop()


st.success("تم تحميل الملف بنجاح.")

col1, col2 = st.columns(2)

with col1:
    st.metric("إجمالي السجلات", len(df))

with col2:
    st.metric("عدد الأعمدة", len(df.columns))


st.subheader("معاينة البيانات")
st.dataframe(df.head(20), use_container_width=True)


st.subheader("اختيار عمود النص")

text_column = st.selectbox(
    "اختاري العمود الذي يحتوي على نص التذكرة أو الشكوى",
    df.columns,
    key="text_column_selector",
)


st.subheader("إعدادات التشغيل")

use_llm = st.checkbox(
    "استخدام LLM Masking المتقدم",
    value=False,
    key="use_llm_masking",
    help="للديمو السريع اتركيه مقفل. فعليه فقط عند الحاجة ولعدد محدود من الصفوف.",
)

max_rows = st.number_input(
    "عدد الصفوف التي تريدين تحليلها الآن",
    min_value=1,
    max_value=len(df),
    value=min(50, len(df)),
    step=10,
    key="max_rows_input",
)


if st.button(
    "تشغيل Privacy & Filtering Agent",
    key="run_privacy_filtering_button",
):
    working_df = df.head(max_rows).copy()

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("جاري تشغيل Privacy & Filtering Agent..."):
        for counter, (index, row) in enumerate(working_df.iterrows(), start=1):
            raw_text = row[text_column]

            status_text.write(
                f"جاري تحليل التذكرة رقم {counter} من {len(working_df)}"
            )

            if use_llm:
                agent_output = run_privacy_filtering_agent(raw_text)
            else:
                agent_output = run_regex_privacy_filtering(raw_text)

            results.append(
                {
                    "ticket_id": index + 1,
                    "raw_text": raw_text,
                    "masked_text": agent_output.get("masked_text", ""),
                    "is_valid_ticket": agent_output.get("is_valid_ticket", False),
                    "message_type": agent_output.get("message_type", "unknown"),
                    "removed_sensitive_items": ", ".join(
                        agent_output.get("removed_sensitive_items", [])
                    ),
                }
            )

            progress_bar.progress(counter / len(working_df))

    output_df = pd.DataFrame(results)
    st.session_state["output_df"] = output_df

    st.success("تم تشغيل Privacy & Filtering Agent بنجاح.")

    valid_count = output_df["is_valid_ticket"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("إجمالي التذاكر المحللة", len(output_df))

    with col2:
        st.metric("التذاكر الصالحة للتحليل", int(valid_count))

    with col3:
        st.metric("التذاكر المستبعدة", int(len(output_df) - valid_count))

    st.subheader("Agent Output Table")
    st.dataframe(output_df, use_container_width=True)

    csv = output_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="تحميل النتائج CSV",
        data=csv,
        file_name="zero_ticket_privacy_output.csv",
        mime="text/csv",
        key="download_privacy_output",
    )


if "output_df" in st.session_state:
    st.divider()
    st.subheader("المرحلة التالية")
    st.write(
        "بعد هذه الخطوة سنضيف Classification Agent لتصنيف المشكلة، مرحلة رحلة المستفيد، ونوع الحل المناسب."
    )