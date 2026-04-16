import json
import os
import random
import html
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path("set1-7.csv")
PROGRESS_PATH = DATA_DIR / "progress.json"

BUCKET_LABELS = {
    0: "List 0 · Unseen / Wrong",
    1: "List 1 · Correct once",
    2: "List 2 · Correct twice",
    3: "List 3 · Correct three times",
}

ANSWER_COLS = ["Antwort_A", "Antwort_B", "Antwort_C", "Antwort_D"]
CORRECT_COLS = ["A_korrekt", "B_korrekt", "C_korrekt", "D_korrekt"]
OPTION_KEYS = ["A", "B", "C", "D"]


@st.cache_data
def load_questions(csv_path_str: str) -> pd.DataFrame:
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=";")

    required_cols = ["Frage", *ANSWER_COLS, *CORRECT_COLS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df.copy().reset_index(drop=True)
    df["question_id"] = df.index.astype(str)
    return df


def build_default_progress(question_ids: list[str]) -> dict:
    shuffled = question_ids[:]
    random.shuffle(shuffled)
    return {
        "bucket_0": shuffled,
        "bucket_1": [],
        "bucket_2": [],
        "bucket_3": [],
    }


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_progress(progress: dict, valid_ids: set[str]) -> dict:
    cleaned = {}
    seen = set()

    for bucket_idx in range(4):
        key = f"bucket_{bucket_idx}"
        items = progress.get(key, [])
        cleaned_bucket = []
        for qid in items:
            qid = str(qid)
            if qid in valid_ids and qid not in seen:
                cleaned_bucket.append(qid)
                seen.add(qid)
        cleaned[key] = cleaned_bucket

    missing = [qid for qid in valid_ids if qid not in seen]
    random.shuffle(missing)
    cleaned["bucket_0"].extend(missing)
    return cleaned


def load_or_create_progress(df: pd.DataFrame) -> dict:
    valid_ids = set(df["question_id"].tolist())

    if not PROGRESS_PATH.exists():
        progress = build_default_progress(df["question_id"].tolist())
        save_progress(progress)
        return progress

    try:
        progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        progress = build_default_progress(df["question_id"].tolist())
        save_progress(progress)
        return progress

    progress = normalize_progress(progress, valid_ids)
    save_progress(progress)
    return progress


def bucket_key(bucket_idx: int) -> str:
    return f"bucket_{bucket_idx}"


def get_question_row(df: pd.DataFrame, qid: str) -> pd.Series:
    row = df.loc[df["question_id"] == str(qid)]
    if row.empty:
        raise KeyError(f"Question ID not found: {qid}")
    return row.iloc[0]


def question_type(row: pd.Series) -> str:
    n_correct = int(sum(int(row[c]) for c in CORRECT_COLS))
    return "Multiple choice" if n_correct > 1 else "Single choice"


def correct_indices(row: pd.Series) -> list[int]:
    return [i for i, col in enumerate(CORRECT_COLS) if int(row[col]) == 1]


def correct_letters_from_indices(indices: list[int]) -> list[str]:
    return [OPTION_KEYS[i] for i in indices]


def correct_answers_text(row: pd.Series, indices: list[int]) -> list[str]:
    return [f"{OPTION_KEYS[i]}: {row[ANSWER_COLS[i]]}" for i in indices]


def remove_from_all_buckets(progress: dict, qid: str) -> None:
    for i in range(4):
        key = bucket_key(i)
        progress[key] = [x for x in progress[key] if x != str(qid)]


def insert_at_position_five(items: list[str], qid: str) -> list[str]:
    idx = min(4, len(items))
    return items[:idx] + [qid] + items[idx:]


def process_answer(progress: dict, current_bucket: int, qid: str, is_correct: bool) -> tuple[str, int]:
    qid = str(qid)
    remove_from_all_buckets(progress, qid)

    if is_correct:
        target_bucket = min(current_bucket + 1, 3)
        progress[bucket_key(target_bucket)].append(qid)
        save_progress(progress)
        return ("correct", target_bucket)

    target_bucket = 0
    progress[bucket_key(0)] = insert_at_position_five(progress[bucket_key(0)], qid)
    save_progress(progress)
    return ("wrong", target_bucket)


def initialize_session_state() -> None:
    defaults = {
        "active_bucket": 0,
        "feedback_ready": False,
        "last_result": None,
        "current_question_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def pick_current_question(progress: dict) -> str | None:
    active_bucket = st.session_state.active_bucket
    queue = progress[bucket_key(active_bucket)]
    if not queue:
        return None
    return queue[0]


def set_next_question(progress: dict) -> None:
    st.session_state.feedback_ready = False
    st.session_state.last_result = None
    st.session_state.current_question_id = pick_current_question(progress)


def switch_bucket(bucket_idx: int, progress: dict) -> None:
    st.session_state.active_bucket = bucket_idx
    set_next_question(progress)
    st.rerun()


def render_bucket_buttons(progress: dict) -> None:
    st.subheader("Choose a list")
    cols = st.columns(4)
    for idx, col in enumerate(cols):
        count = len(progress[bucket_key(idx)])
        label = f"{BUCKET_LABELS[idx]}\n({count})"
        if col.button(label, use_container_width=True, key=f"bucket_btn_{idx}"):
            switch_bucket(idx, progress)


def reset_progress(df: pd.DataFrame) -> None:
    progress = build_default_progress(df["question_id"].tolist())
    save_progress(progress)
    st.session_state.active_bucket = 0
    set_next_question(progress)
    st.rerun()


def render_question(row: pd.Series, q_type: str) -> None:
    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:30px; line-height:1.5; font-weight:600;">
            {html.escape(str(row['Frage']))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(q_type)


def main() -> None:
    st.set_page_config(page_title="Flashcard Trainer", page_icon="🧠", layout="wide")
    st.title("🧠 Flashcard Trainer")

    try:
        df = load_questions(str(CSV_PATH))
    except Exception as exc:
        st.error(str(exc))
        st.info("Place 'set1-7.csv' in the same folder as app.py.")
        st.stop()

    progress = load_or_create_progress(df)
    initialize_session_state()

    if st.session_state.current_question_id is None:
        st.session_state.current_question_id = pick_current_question(progress)

    with st.sidebar:
        st.header("Progress")
        for idx in range(4):
            st.write(f"**{BUCKET_LABELS[idx]}:** {len(progress[bucket_key(idx)])}")
        st.divider()
        if st.button("Reset all progress", type="secondary", use_container_width=True):
            reset_progress(df)
        st.caption("Progress is stored locally in progress.json.")

    render_bucket_buttons(progress)

    active_bucket = st.session_state.active_bucket
    active_key = bucket_key(active_bucket)
    active_queue = progress[active_key]

    st.markdown(f"### Current list: {BUCKET_LABELS[active_bucket]}")

    if not active_queue:
        st.warning("This list is empty. Choose another list above.")
        st.session_state.current_question_id = None
        st.stop()

    current_qid = st.session_state.current_question_id

    if current_qid is None:
        current_qid = pick_current_question(progress)
        st.session_state.current_question_id = current_qid

    if current_qid is None:
        st.warning("This list is empty. Choose another list above.")
        st.stop()

    row = get_question_row(df, current_qid)
    q_type = question_type(row)
    correct_idx = correct_indices(row)
    correct_set = set(correct_idx)

    total_in_bucket = len(active_queue)
    st.caption(f"Questions currently in this list: {total_in_bucket}")

    render_question(row, q_type)

    if not st.session_state.feedback_ready:
        with st.form(key=f"question_form_{current_qid}"):
            selected_indices = []

            if q_type == "Single choice":
                options = [f"{OPTION_KEYS[i]}: {row[ANSWER_COLS[i]]}" for i in range(4)]
                choice = st.radio(
                    "Choose one answer:",
                    options=range(4),
                    format_func=lambda i: options[i],
                    index=None,
                    key=f"radio_{current_qid}",
                )
                if choice is not None:
                    selected_indices = [choice]
            else:
                st.write("Choose one or more answers:")
                for i in range(4):
                    checked = st.checkbox(
                        f"{OPTION_KEYS[i]}: {row[ANSWER_COLS[i]]}",
                        value=False,
                        key=f"check_{current_qid}_{i}",
                    )
                    if checked:
                        selected_indices.append(i)

            submitted = st.form_submit_button("Submit", use_container_width=True)

        if submitted:
            selected_set = set(selected_indices)
            is_correct = selected_set == correct_set
            result, target_bucket = process_answer(progress, active_bucket, current_qid, is_correct)

            st.session_state.feedback_ready = True
            st.session_state.last_result = {
                "question_id": current_qid,
                "selected_indices": selected_indices,
                "is_correct": is_correct,
                "result": result,
                "moved_to_bucket": target_bucket,
                "correct_indices": correct_idx,
                "bucket_when_answered": active_bucket,
            }
            st.rerun()

    else:
        result = st.session_state.last_result

        if not result or result["question_id"] != current_qid:
            st.session_state.feedback_ready = False
            st.session_state.last_result = None
            st.rerun()

        st.markdown("---")

        if result["is_correct"]:
            st.success(
                f"Correct. The question was moved to {BUCKET_LABELS[result['moved_to_bucket']]}."
            )
        else:
            st.error(
                f"Incorrect. The question was moved to {BUCKET_LABELS[result['moved_to_bucket']]} and will reappear soon."
            )

        selected_letters = correct_letters_from_indices(result["selected_indices"])
        correct_letters_list = correct_letters_from_indices(result["correct_indices"])

        st.write(f"**Your answer:** {', '.join(selected_letters) if selected_letters else 'No answer selected'}")
        st.write(f"**Correct answer(s):** {', '.join(correct_letters_list)}")

        st.write("**Correct option text:**")
        for text in correct_answers_text(row, result["correct_indices"]):
            st.write(f"- {text}")

        st.write("**All answer options:**")
        for i in range(4):
            marker = "✅" if i in correct_set else "❌"
            selected_marker = " ← your choice" if i in result["selected_indices"] else ""
            st.write(f"{marker} {OPTION_KEYS[i]}: {row[ANSWER_COLS[i]]}{selected_marker}")

        if st.button("Next question", type="primary", use_container_width=True):
            updated_progress = load_or_create_progress(df)
            set_next_question(updated_progress)
            st.rerun()


if __name__ == "__main__":
    main()
