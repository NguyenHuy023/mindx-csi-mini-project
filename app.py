import json
import os
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "gemini-3.6-flash"


def get_streamlit_secret(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data
def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_tour() -> pd.DataFrame:
    return pd.read_csv("tour.csv", index_col=0)


def load_api_key() -> Optional[str]:
    load_dotenv()
    return (
        os.getenv("GEMINI_API_KEY")
        or get_streamlit_secret("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or get_streamlit_secret("GOOGLE_API_KEY")
    )


@st.cache_resource
def create_client(api_key: Optional[str]):
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def build_tour_context(menu_df: pd.DataFrame) -> str:
    def clean_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    def add_detail(item_text: str, label: str, value) -> str:
        value = clean_text(value)
        if not value:
            return item_text

        suffix = "" if value.endswith((".", "!", "?")) else "."
        return f"{item_text} {label}: {value}{suffix}"

    menu_lines = []
    for _, row in menu_df.iterrows():
        item_text = f"- {clean_text(row['tour_name'])}: {clean_text(row['itinerary'])}"
        item_text = add_detail(item_text, "bao gồm", row.get("inclusions", ""))
        item_text = add_detail(item_text, "thời gian", row.get("departure", ""))
        menu_lines.append(item_text)
    return "\n".join(menu_lines)


def build_system_instruction(config: dict) -> str:
    functions = ", ".join(config.get("functions", []))
    agency_name = config.get("agency_name", "Việt Nam du lịch")
    agency_address = config.get("agency_address", "190 Pasteur, Quận 3, TP.HCM")
    out_of_scope_message = config.get("out_of_scope_message", "Hiện tại tôi chưa có tour này. Bạn muốn tìm địa điểm nào khác không?")
    return "\n".join([
        f"Bạn tên là TourBot, một trợ lý AI hỗ trợ khách hàng của công ti du lịch {agency_name}.",
        f"Địa chỉ công ti: {agency_address}.",
        f"Các chức năng được hỗ trợ: {functions}.",
        "",
        "Nguyên tắc trả lời:",
        "1. Trả lời ngắn gọn, thân thiện, lịch sự và dễ hiểu.",
        "2. Chỉ trả lời các câu hỏi liên quan đến dịch vụ du lịch Việt Nam.",
        f"3. Nếu câu hỏi nằm ngoài phạm vi hỗ trợ, trả lời đúng câu sau: \"{out_of_scope_message}\"",
        "4. Không bịa thông tin nếu dữ liệu không có câu trả lời.",
    ])


def build_history_text(messages: list[dict], max_messages: int = 10) -> str:
    recent_messages = messages[-max_messages:]
    lines = []
    for message in recent_messages:
        role = "Khách hàng" if message["role"] == "user" else "TourBot"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def mock_response(prompt: str, menu_df: pd.DataFrame, config: dict) -> str:
    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in ["menu", "món", "mon", "ăn", "an", "food"]):
        return "\n\n".join(
            f"**{row['name']}**: {row['description']}" for _, row in menu_df.iterrows()
        )
    if any(keyword in prompt_lower for keyword in ["địa chỉ", "dia chi", "ở đâu", "address"]):
        return f"Nhà hàng {config.get('restaurant_name')} nằm tại {config.get('restaurant_address')}."
    if any(keyword in prompt_lower for keyword in ["xin chào", "hello", "hi", "chào"]):
        return "Chào bạn! Tôi là PhoBot. Bạn có thể hỏi tôi về nhà hàng hoặc các món trong menu nhé."
    return config.get("out_of_scope_message")


def ask_bot(prompt: str, messages: list[dict], client, tour_df: pd.DataFrame, config: dict, use_mock: bool) -> str:
    if use_mock or client is None:
        return mock_response(prompt, tour_df, config)

    system_instruction = build_system_instruction(config)
    tour_context = build_tour_context(tour_df)
    history_text = build_history_text(messages)
    user_content = (
        "Dữ liệu danh mục các tours trong công ti:\n" + tour_context +
        "\n\nLịch sử trò chuyện gần đây:\n" + history_text +
        "\n\nCâu hỏi mới của khách hàng:\n" + prompt
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            ),
        )
        return response.text or "Xin lỗi, tôi chưa tạo được câu trả lời phù hợp."
    except Exception as exc:
        return (
            "Xin lỗi, hiện tại tôi chưa kết nối được với Gemini API. "
            "Bạn có thể kiểm tra lại API key, kết nối mạng hoặc dùng chế độ Mock để tiếp tục demo.\n\n"
            f"Chi tiết lỗi: `{exc}`"
        )


def travel_chatbot():
    config = load_config()
    tour_df = load_tour()
    api_key = load_api_key()
    client = create_client(api_key)

    st.set_page_config(page_title="TourBot - Tour Assistant", page_icon="🍜")
    st.title("🍜 TourBot - Tour Assistant")
    st.write("Trợ lý ảo hỗ trợ khách hàng tìm kiếm tours.")

    with st.sidebar:
        st.header("Cấu hình")
        st.caption("Dùng cho buổi 8: Streamlit Chat UI + Gemini API")
        has_sdk = genai is not None
        st.write("Google GenAI SDK:", "✅ Đã sẵn sàng" if has_sdk else "⚠️ Chưa cài `google-genai`")
        st.write("API key:", "✅ Đã tìm thấy" if api_key else "⚠️ Chưa có")
        use_mock = st.toggle(
            "Dùng Mock Response",
            value=not bool(api_key and has_sdk),
            help="Bật chế độ này nếu học viên chưa tạo được API key hoặc lớp cần tránh lỗi API.",
        )
        st.markdown("---")
        st.markdown("**Gợi ý câu hỏi:**")
        st.markdown("- Công ti du lịch nằm ở đâu?")
        st.markdown("- Tìm cho tôi tour du lịch ở Miền Bắc")
        st.markdown("- Tìm cho tôi tour du lịch ở tam đảo")
        st.markdown("- Có tour nào ở Bà Nà hill không?")

    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = [
            {"role": "assistant", "content": config.get("initial_bot_message", "Xin chào! Bạn cần hỗ trợ gì?")}
        ]

    for message in st.session_state.conversation_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Nhập yêu cầu của bạn tại đây...")
    if prompt:
        st.session_state.conversation_log.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        bot_reply = ask_bot(
            prompt=prompt,
            messages=st.session_state.conversation_log,
            client=client,
            tour_df=tour_df,
            config=config,
            use_mock=use_mock,
        )

        st.session_state.conversation_log.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.write(bot_reply)

    with st.expander("Checklist bảo mật API key"):
        st.markdown(
            """
- Không dán API key trực tiếp vào `app.py`.
- Local: lưu key trong `.env` với tên `GEMINI_API_KEY`.
- Deploy Streamlit Cloud: lưu key trong **Secrets**.
- Không commit file `.env` lên GitHub.
- Nếu thiếu key, app vẫn có thể chạy bằng Mock Response để demo giao diện.
"""
        )


if __name__ == "__main__":
    travel_chatbot()
