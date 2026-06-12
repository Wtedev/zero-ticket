
import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Zero Ticket AI Agent",
    page_icon="🎫",
    layout="wide"
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
    type=["xlsx", "csv"]
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


st.subheader("أسماء الأعمدة")
st.write(list(df.columns))


st.divider()

st.subheader("الخطوة التالية")
st.write(
    "بعد التأكد من قراءة الملف، سنبني Privacy & Filtering Agent لإخفاء البيانات الحساسة وتنظيف النصوص."
)