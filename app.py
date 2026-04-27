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
    0: "List 0 · Unseen / Incorrect",
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


def letters_from_indices(indices: list[int]) -> list[str]:
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


def pick_current_question(progress: dict, bucket_idx: int) -> str | None:
    queue = progress[bucket_key(bucket_idx)]
    if not queue:
        return None
    return queue[0]


def switch_bucket(bucket_idx: int, progress: dict) -> None:
    st.session_state.active_bucket = bucket_idx
    st.session_state.mode = "question"
    st.session_state.last_result = None
    st.session_state.current_question_id = pick_current_question(progress, bucket_idx)
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
    st.session_state.mode = "question"
    st.session_state.last_result = None
    st.session_state.current_question_id = pick_current_question(progress, 0)
    st.rerun()


def render_question_text(row: pd.Series, q_type: str) -> None:
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


def clear_question_widget_state(prefix: str) -> None:
    keys = [
        f"{prefix}_radio",
        f"{prefix}_check_0",
        f"{prefix}_check_1",
        f"{prefix}_check_2",
        f"{prefix}_check_3",
    ]
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def initialize_session_state(progress: dict) -> None:
    defaults = {
        "app_mode": "training",
        "active_bucket": 0,
        "mode": "question",
        "current_question_id": pick_current_question(progress, 0),
        "last_result": None,
        "test_question_count": 20,
        "test_questions": [],
        "test_index": 0,
        "test_answers": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_test(df: pd.DataFrame, n_questions: int) -> None:
    all_qids = df["question_id"].tolist()
    if not all_qids:
        st.error("No questions available for the test.")
        return

    n_questions = min(n_questions, len(all_qids))
    sampled = random.sample(all_qids, n_questions)

    st.session_state.test_questions = sampled
    st.session_state.test_index = 0
    st.session_state.test_answers = {}
    st.session_state.app_mode = "test_run"
    st.rerun()


def render_answer_inputs(prefix: str, row: pd.Series, q_type: str, disabled: bool = False) -> list[int]:
    selected_indices = []

    if q_type == "Single choice":
        options = [f"{OPTION_KEYS[i]}: {row[ANSWER_COLS[i]]}" for i in range(4)]
        choice = st.radio(
            "Choose one answer:",
            options=range(4),
            format_func=lambda i: options[i],
            index=None,
            key=f"{prefix}_radio",
            disabled=disabled,
        )
        if choice is not None:
            selected_indices = [choice]
    else:
        st.write("Choose one or more answers:")
        for i in range(4):
            checked = st.checkbox(
                f"{OPTION_KEYS[i]}: {row[ANSWER_COLS[i]]}",
                value=False,
                key=f"{prefix}_check_{i}",
                disabled=disabled,
            )
            if checked:
                selected_indices.append(i)

    return selected_indices


def render_training_mode(df: pd.DataFrame, progress: dict) -> None:
    render_bucket_buttons(progress)

    active_bucket = st.session_state.active_bucket
    active_queue = progress[bucket_key(active_bucket)]

    st.markdown(f"### Current list: {BUCKET_LABELS[active_bucket]}")

    if not active_queue and st.session_state.mode == "question":
        st.warning("This list is empty. Choose another list above.")
        st.stop()

    current_qid = st.session_state.current_question_id
    if current_qid is None:
        current_qid = pick_current_question(progress, active_bucket)
        st.session_state.current_question_id = current_qid

    if current_qid is None:
        st.warning("This list is empty. Choose another list above.")
        st.stop()

    row = get_question_row(df, current_qid)
    q_type = question_type(row)
    correct_idx = correct_indices(row)
    correct_set = set(correct_idx)

    st.caption(f"Questions currently in this list: {len(active_queue)}")
    render_question_text(row, q_type)

    if st.session_state.mode == "question":
        selected_indices = render_answer_inputs(prefix=f"train_{current_qid}", row=row, q_type=q_type)

        if st.button("Submit", type="primary", use_container_width=True, key=f"submit_{current_qid}"):
            selected_set = set(selected_indices)
            is_correct = selected_set == correct_set
            _, target_bucket = process_answer(progress, active_bucket, current_qid, is_correct)

            st.session_state.last_result = {
                "question_id": current_qid,
                "selected_indices": selected_indices,
                "is_correct": is_correct,
                "moved_to_bucket": target_bucket,
                "correct_indices": correct_idx,
            }
            st.session_state.mode = "feedback"
            st.rerun()

    else:
        result = st.session_state.last_result

        if not result or result["question_id"] != current_qid:
            st.session_state.mode = "question"
            st.session_state.last_result = None
            st.rerun()

        st.markdown("---")

        if result["is_correct"]:
            st.success(f"Correct. The question was moved to {BUCKET_LABELS[result['moved_to_bucket']]}.")
        else:
            st.error(
                f"Incorrect. The question was moved to {BUCKET_LABELS[result['moved_to_bucket']]} and will appear again soon."
            )

        selected_letters = letters_from_indices(result["selected_indices"])
        correct_letters_list = letters_from_indices(result["correct_indices"])

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

        if st.button("Next question", type="primary", use_container_width=True, key=f"next_{current_qid}"):
            updated_progress = load_or_create_progress(df)
            next_qid = pick_current_question(updated_progress, st.session_state.active_bucket)

            clear_question_widget_state(f"train_{current_qid}")
            st.session_state.last_result = None
            st.session_state.mode = "question"
            st.session_state.current_question_id = next_qid
            st.rerun()


def render_test_setup(df: pd.DataFrame) -> None:
    st.markdown("## Test mode")
    st.write("Choose how many random questions you want in your test.")

    max_questions = len(df)
    n_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=max_questions,
        value=min(st.session_state.test_question_count, max_questions),
        step=1,
    )
    st.session_state.test_question_count = int(n_questions)

    if st.button("Start test", type="primary", use_container_width=True):
        start_test(df, int(n_questions))


def render_test_run(df: pd.DataFrame) -> None:
    test_questions = st.session_state.test_questions
    idx = st.session_state.test_index

    if idx >= len(test_questions):
        st.session_state.app_mode = "test_result"
        st.rerun()

    current_qid = test_questions[idx]
    row = get_question_row(df, current_qid)
    q_type = question_type(row)
    correct_idx = correct_indices(row)
    correct_set = set(correct_idx)

    st.markdown("## Test mode")
    st.caption(f"Question {idx + 1} of {len(test_questions)}")
    render_question_text(row, q_type)

    selected_indices = render_answer_inputs(prefix=f"test_{current_qid}", row=row, q_type=q_type)

    if st.button("Submit answer", type="primary", use_container_width=True, key=f"test_submit_{current_qid}"):
        selected_set = set(selected_indices)
        is_correct = selected_set == correct_set

        st.session_state.test_answers[current_qid] = {
            "selected_indices": selected_indices,
            "correct_indices": correct_idx,
            "is_correct": is_correct,
        }

        clear_question_widget_state(f"test_{current_qid}")
        st.session_state.test_index += 1

        if st.session_state.test_index >= len(test_questions):
            st.session_state.app_mode = "test_result"
        else:
            st.session_state.app_mode = "test_run"

        st.rerun()


def render_test_result(df: pd.DataFrame) -> None:
    test_questions = st.session_state.test_questions
    test_answers = st.session_state.test_answers

    total = len(test_questions)
    correct = sum(1 for qid in test_questions if test_answers.get(qid, {}).get("is_correct", False))
    wrong = total - correct
    percentage = (correct / total * 100) if total > 0 else 0.0

    st.markdown("## Test result")
    st.success("Test finished.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Correct", correct)
    c2.metric("Incorrect", wrong)
    c3.metric("Percentage", f"{percentage:.1f}%")

    with st.expander("Show review"):
        for i, qid in enumerate(test_questions, start=1):
            row = get_question_row(df, qid)
            result = test_answers.get(qid, {})
            is_correct = result.get("is_correct", False)
            selected = result.get("selected_indices", [])
            correct_idx = result.get("correct_indices", [])

            st.markdown(f"### {i}. {row['Frage']}")
            st.write(f"**Result:** {'✅ Correct' if is_correct else '❌ Incorrect'}")
            st.write(f"**Your answer:** {', '.join(letters_from_indices(selected)) if selected else 'No answer selected'}")
            st.write(f"**Correct answer(s):** {', '.join(letters_from_indices(correct_idx))}")

    if st.button("Start new test", use_container_width=True):
        st.session_state.app_mode = "test_setup"
        st.session_state.test_questions = []
        st.session_state.test_index = 0
        st.session_state.test_answers = {}
        st.rerun()

    if st.button("Back to training", use_container_width=True):
        st.session_state.app_mode = "training"
        st.session_state.test_questions = []
        st.session_state.test_index = 0
        st.session_state.test_answers = {}
        st.rerun()


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
    initialize_session_state(progress)

    with st.sidebar:
        st.header("Mode")
        if st.button("Training mode", use_container_width=True):
            st.session_state.app_mode = "training"
            st.rerun()

        if st.button("Test mode", use_container_width=True):
            st.session_state.app_mode = "test_setup"
            st.rerun()

        st.divider()

        st.header("Progress")
        for idx in range(4):
            st.write(f"**{BUCKET_LABELS[idx]}:** {len(progress[bucket_key(idx)])}")
        st.divider()

        if st.button("Reset all progress", type="secondary", use_container_width=True):
            reset_progress(df)

        st.caption("Progress is stored locally in progress.json.")

    if st.session_state.app_mode == "training":
        render_training_mode(df, progress)
    elif st.session_state.app_mode == "test_setup":
        render_test_setup(df)
    elif st.session_state.app_mode == "test_run":
        render_test_run(df)
    elif st.session_state.app_mode == "test_result":
        render_test_result(df)


if __name__ == "__main__":
    main()
