import json
import os
import random
import html
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "train4": {
        "label": "Training · 4 Antworten",
        "csv_path": Path("set1-7.csv"),
        "progress_path": DATA_DIR / "progress_4.json",
        "option_keys": ["A", "B", "C", "D"],
        "answer_cols": ["Antwort_A", "Antwort_B", "Antwort_C", "Antwort_D"],
        "correct_cols": ["A_korrekt", "B_korrekt", "C_korrekt", "D_korrekt"],
    },
    "train5": {
        "label": "Training · 5 Antworten",
        "csv_path": Path("set5.csv"),
        "progress_path": DATA_DIR / "progress_5.json",
        "option_keys": ["A", "B", "C", "D", "E"],
        "answer_cols": ["Antwort_A", "Antwort_B", "Antwort_C", "Antwort_D", "Antwort_E"],
        "correct_cols": ["A_korrekt", "B_korrekt", "C_korrekt", "D_korrekt", "E_korrekt"],
    },
}

BUCKET_LABELS = {
    0: "List 0 · Ungesehen / Falsch",
    1: "List 1 · 1x richtig",
    2: "List 2 · 2x richtig",
    3: "List 3 · 3x richtig",
}


@st.cache_data
def load_questions(csv_path_str: str, answer_cols: tuple, correct_cols: tuple) -> pd.DataFrame:
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.resolve()}")

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=";")

    required_cols = ["Frage", *list(answer_cols), *list(correct_cols)]
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
        "bucket_0": [],
        "bucket_1": shuffled,
        "bucket_2": [],
        "bucket_3": [],
    }

def save_progress(progress: dict, progress_path: Path) -> None:
    progress_path.write_text(
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
    cleaned["bucket_1"].extend(missing)
    return cleaned


def load_or_create_progress(df: pd.DataFrame, progress_path: Path) -> dict:
    valid_ids = set(df["question_id"].tolist())

    if not progress_path.exists():
        progress = build_default_progress(df["question_id"].tolist())
        save_progress(progress, progress_path)
        return progress

    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        progress = build_default_progress(df["question_id"].tolist())
        save_progress(progress, progress_path)
        return progress

    progress = normalize_progress(progress, valid_ids)
    save_progress(progress, progress_path)
    return progress


def bucket_key(bucket_idx: int) -> str:
    return f"bucket_{bucket_idx}"


def get_question_row(df: pd.DataFrame, qid: str) -> pd.Series:
    row = df.loc[df["question_id"] == str(qid)]
    if row.empty:
        raise KeyError(f"Question ID not found: {qid}")
    return row.iloc[0]


def question_type(row: pd.Series, correct_cols: list[str]) -> str:
    n_correct = int(sum(int(row[c]) for c in correct_cols))
    return "Multiple choice" if n_correct > 1 else "Single choice"


def correct_indices(row: pd.Series, correct_cols: list[str]) -> list[int]:
    return [i for i, col in enumerate(correct_cols) if int(row[col]) == 1]


def letters_from_indices(indices: list[int], option_keys: list[str]) -> list[str]:
    return [option_keys[i] for i in indices]


def correct_answers_text(row: pd.Series, indices: list[int], answer_cols: list[str], option_keys: list[str]) -> list[str]:
    return [f"{option_keys[i]}: {row[answer_cols[i]]}" for i in indices]


def remove_from_all_buckets(progress: dict, qid: str) -> None:
    for i in range(4):
        key = bucket_key(i)
        progress[key] = [x for x in progress[key] if x != str(qid)]


def insert_at_position_five(items: list[str], qid: str) -> list[str]:
    idx = min(4, len(items))
    return items[:idx] + [qid] + items[idx:]


def process_answer(progress: dict, current_bucket: int, qid: str, is_correct: bool, progress_path: Path) -> tuple[str, int]:
    qid = str(qid)
    remove_from_all_buckets(progress, qid)

    if is_correct:
        target_bucket = min(current_bucket + 1, 3)
        progress[bucket_key(target_bucket)].append(qid)
        save_progress(progress, progress_path)
        return ("correct", target_bucket)

    target_bucket = 0
    progress[bucket_key(0)] = insert_at_position_five(progress[bucket_key(0)], qid)
    save_progress(progress, progress_path)
    return ("wrong", target_bucket)


def move_question_to_bucket0(progress: dict, qid: str, progress_path: Path) -> None:
    qid = str(qid)
    remove_from_all_buckets(progress, qid)
    progress[bucket_key(0)] = insert_at_position_five(progress[bucket_key(0)], qid)
    save_progress(progress, progress_path)


def pick_current_question(progress: dict, bucket_idx: int) -> str | None:
    queue = progress[bucket_key(bucket_idx)]
    if not queue:
        return None
    return queue[0]


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


def clear_question_widget_state(qid: str, option_count: int) -> None:
    keys = [f"radio_{qid}"]
    for i in range(option_count):
        keys.append(f"check_{qid}_{i}")
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def initialize_session_state() -> None:
    defaults = {
        "app_mode": "train4",  # train4 | train5 | selftest
        "train4_active_bucket": 0,
        "train4_screen_mode": "question",
        "train4_current_question_id": None,
        "train4_last_result": None,
        "train5_active_bucket": 0,
        "train5_screen_mode": "question",
        "train5_current_question_id": None,
        "train5_last_result": None,
        "test_num_questions": 20,
        "test_started": False,
        "test_questions": [],
        "test_current_index": 0,
        "test_answers": {},
        "test_show_feedback": False,
        "test_finished": False,
        "test_progress_applied": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_training_state_keys(dataset_key: str) -> dict:
    return {
        "active_bucket": f"{dataset_key}_active_bucket",
        "screen_mode": f"{dataset_key}_screen_mode",
        "current_question_id": f"{dataset_key}_current_question_id",
        "last_result": f"{dataset_key}_last_result",
    }


def ensure_current_question_for_dataset(dataset_key: str, progress: dict) -> None:
    keys = get_training_state_keys(dataset_key)
    if st.session_state[keys["current_question_id"]] is None:
        bucket_idx = st.session_state[keys["active_bucket"]]
        st.session_state[keys["current_question_id"]] = pick_current_question(progress, bucket_idx)


def switch_training_bucket(dataset_key: str, bucket_idx: int, progress: dict) -> None:
    keys = get_training_state_keys(dataset_key)
    st.session_state[keys["active_bucket"]] = bucket_idx
    st.session_state[keys["screen_mode"]] = "question"
    st.session_state[keys["last_result"]] = None
    st.session_state[keys["current_question_id"]] = pick_current_question(progress, bucket_idx)
    st.rerun()


def render_training_bucket_buttons(dataset_key: str, progress: dict) -> None:
    keys = get_training_state_keys(dataset_key)
    st.subheader("Listen auswählen")
    cols = st.columns(4)
    for idx, col in enumerate(cols):
        count = len(progress[bucket_key(idx)])
        label = f"{BUCKET_LABELS[idx]}\n({count})"
        if col.button(label, use_container_width=True, key=f"{dataset_key}_bucket_btn_{idx}"):
            switch_training_bucket(dataset_key, idx, progress)


def reset_progress_for_dataset(dataset_key: str, df: pd.DataFrame) -> None:
    config = DATASETS[dataset_key]
    progress = build_default_progress(df["question_id"].tolist())
    save_progress(progress, config["progress_path"])

    keys = get_training_state_keys(dataset_key)
    st.session_state[keys["active_bucket"]] = 0
    st.session_state[keys["screen_mode"]] = "question"
    st.session_state[keys["last_result"]] = None
    st.session_state[keys["current_question_id"]] = pick_current_question(progress, 0)
    st.rerun()


def load_all_data():
    dfs = {}
    progresses = {}

    for dataset_key, config in DATASETS.items():
        df = load_questions(
            str(config["csv_path"]),
            tuple(config["answer_cols"]),
            tuple(config["correct_cols"]),
        )
        progress = load_or_create_progress(df, config["progress_path"])
        dfs[dataset_key] = df
        progresses[dataset_key] = progress

    return dfs, progresses


def render_sidebar(dfs: dict, progresses: dict) -> None:
    st.sidebar.title("Navigation")

    if st.sidebar.button("Training · 4 Antworten", use_container_width=True):
        st.session_state.app_mode = "train4"
        st.rerun()

    if st.sidebar.button("Training · 5 Antworten", use_container_width=True):
        st.session_state.app_mode = "train5"
        st.rerun()

    if st.sidebar.button("Selbsttest", use_container_width=True):
        st.session_state.app_mode = "selftest"
        st.rerun()

    st.sidebar.divider()

    st.sidebar.subheader("Fortschritt 4 Antworten")
    for idx in range(4):
        st.sidebar.write(f"**{BUCKET_LABELS[idx]}:** {len(progresses['train4'][bucket_key(idx)])}")

    if st.sidebar.button("Reset 4-Antwort-Fortschritt", use_container_width=True):
        reset_progress_for_dataset("train4", dfs["train4"])

    st.sidebar.divider()

    st.sidebar.subheader("Fortschritt 5 Antworten")
    for idx in range(4):
        st.sidebar.write(f"**{BUCKET_LABELS[idx]}:** {len(progresses['train5'][bucket_key(idx)])}")

    if st.sidebar.button("Reset 5-Antwort-Fortschritt", use_container_width=True):
        reset_progress_for_dataset("train5", dfs["train5"])

    st.sidebar.divider()
    st.sidebar.caption("Progress wird lokal gespeichert in progress_4.json und progress_5.json.")


def render_answer_inputs(prefix: str, row: pd.Series, q_type: str, answer_cols: list[str], option_keys: list[str], disabled: bool = False) -> list[int]:
    selected_indices = []

    if q_type == "Single choice":
        options = [f"{option_keys[i]}: {row[answer_cols[i]]}" for i in range(len(option_keys))]
        choice = st.radio(
            "Wähle eine Antwort:",
            options=range(len(option_keys)),
            format_func=lambda i: options[i],
            index=None,
            key=f"{prefix}_radio",
            disabled=disabled,
        )
        if choice is not None:
            selected_indices = [choice]
    else:
        st.write("Wähle eine oder mehrere Antworten:")
        for i in range(len(option_keys)):
            checked = st.checkbox(
                f"{option_keys[i]}: {row[answer_cols[i]]}",
                value=False,
                key=f"{prefix}_check_{i}",
                disabled=disabled,
            )
            if checked:
                selected_indices.append(i)

    return selected_indices


def render_training_mode(dataset_key: str, df: pd.DataFrame, progress: dict) -> None:
    config = DATASETS[dataset_key]
    keys = get_training_state_keys(dataset_key)

    ensure_current_question_for_dataset(dataset_key, progress)

    active_bucket = st.session_state[keys["active_bucket"]]
    active_queue = progress[bucket_key(active_bucket)]

    st.markdown(f"## {config['label']}")
    render_training_bucket_buttons(dataset_key, progress)
    st.markdown(f"### Aktuelle Liste: {BUCKET_LABELS[active_bucket]}")

    if not active_queue and st.session_state[keys["screen_mode"]] == "question":
        st.warning("Diese Liste ist leer. Bitte wähle eine andere Liste.")
        st.stop()

    current_qid = st.session_state[keys["current_question_id"]]
    if current_qid is None:
        current_qid = pick_current_question(progress, active_bucket)
        st.session_state[keys["current_question_id"]] = current_qid

    if current_qid is None:
        st.warning("Diese Liste ist leer. Bitte wähle eine andere Liste.")
        st.stop()

    row = get_question_row(df, current_qid)
    q_type = question_type(row, config["correct_cols"])
    correct_idx = correct_indices(row, config["correct_cols"])
    correct_set = set(correct_idx)

    st.caption(f"Fragen in dieser Liste: {len(active_queue)}")
    render_question_text(row, q_type)

    if st.session_state[keys["screen_mode"]] == "question":
        selected_indices = render_answer_inputs(
            prefix=f"{dataset_key}_{current_qid}",
            row=row,
            q_type=q_type,
            answer_cols=config["answer_cols"],
            option_keys=config["option_keys"],
            disabled=False,
        )

        if st.button("Submit", type="primary", use_container_width=True, key=f"{dataset_key}_submit_{current_qid}"):
            selected_set = set(selected_indices)
            is_correct = selected_set == correct_set
            _, target_bucket = process_answer(
                progress,
                active_bucket,
                current_qid,
                is_correct,
                config["progress_path"],
            )

            st.session_state[keys["last_result"]] = {
                "question_id": current_qid,
                "selected_indices": selected_indices,
                "is_correct": is_correct,
                "moved_to_bucket": target_bucket,
                "correct_indices": correct_idx,
            }
            st.session_state[keys["screen_mode"]] = "feedback"
            st.rerun()

    else:
        result = st.session_state[keys["last_result"]]

        if not result or result["question_id"] != current_qid:
            st.session_state[keys["screen_mode"]] = "question"
            st.session_state[keys["last_result"]] = None
            st.rerun()

        st.markdown("---")

        if result["is_correct"]:
            st.success(f"Richtig. Die Frage wurde nach {BUCKET_LABELS[result['moved_to_bucket']]} verschoben.")
        else:
            st.error(f"Falsch. Die Frage wurde nach {BUCKET_LABELS[result['moved_to_bucket']]} verschoben und erscheint bald wieder.")

        selected_letters = letters_from_indices(result["selected_indices"], config["option_keys"])
        correct_letters_list = letters_from_indices(result["correct_indices"], config["option_keys"])

        st.write(f"**Deine Antwort:** {', '.join(selected_letters) if selected_letters else 'Keine Antwort ausgewählt'}")
        st.write(f"**Richtige Antwort(en):** {', '.join(correct_letters_list)}")

        st.write("**Richtige Antworttexte:**")
        for text in correct_answers_text(row, result["correct_indices"], config["answer_cols"], config["option_keys"]):
            st.write(f"- {text}")

        st.write("**Alle Antwortoptionen:**")
        for i in range(len(config["option_keys"])):
            marker = "✅" if i in correct_set else "❌"
            selected_marker = " ← deine Auswahl" if i in result["selected_indices"] else ""
            st.write(f"{marker} {config['option_keys'][i]}: {row[config['answer_cols'][i]]}{selected_marker}")

        if st.button("Nächste Frage", type="primary", use_container_width=True, key=f"{dataset_key}_next_{current_qid}"):
            updated_progress = load_or_create_progress(df, config["progress_path"])
            next_qid = pick_current_question(updated_progress, st.session_state[keys["active_bucket"]])

            clear_question_widget_state(f"{dataset_key}_{current_qid}", len(config["option_keys"]))
            st.session_state[keys["last_result"]] = None
            st.session_state[keys["screen_mode"]] = "question"
            st.session_state[keys["current_question_id"]] = next_qid
            st.rerun()


def build_test_pool(dfs: dict) -> list[dict]:
    pool = []
    for dataset_key, df in dfs.items():
        for qid in df["question_id"].tolist():
            pool.append({"dataset_key": dataset_key, "question_id": str(qid)})
    return pool


def reset_test_state() -> None:
    st.session_state.test_started = False
    st.session_state.test_questions = []
    st.session_state.test_current_index = 0
    st.session_state.test_answers = {}
    st.session_state.test_show_feedback = False
    st.session_state.test_finished = False
    st.session_state.test_progress_applied = False


def start_test(dfs: dict, n_questions: int) -> None:
    pool = build_test_pool(dfs)

    if not pool:
        st.error("Keine Fragen für den Test gefunden.")
        return

    n_questions = min(n_questions, len(pool))
    random.shuffle(pool)

    st.session_state.test_questions = pool[:n_questions]
    st.session_state.test_current_index = 0
    st.session_state.test_answers = {}
    st.session_state.test_show_feedback = False
    st.session_state.test_finished = False
    st.session_state.test_progress_applied = False
    st.session_state.test_started = True
    st.rerun()


def apply_test_wrong_answers(progresses: dict) -> None:
    if st.session_state.test_progress_applied:
        return

    wrong_by_dataset = {"train4": set(), "train5": set()}

    for answer_data in st.session_state.test_answers.values():
        if not answer_data["is_correct"]:
            wrong_by_dataset[answer_data["dataset_key"]].add(answer_data["question_id"])

    for dataset_key, qids in wrong_by_dataset.items():
        if not qids:
            continue
        config = DATASETS[dataset_key]
        progress = progresses[dataset_key]
        for qid in qids:
            move_question_to_bucket0(progress, qid, config["progress_path"])

    st.session_state.test_progress_applied = True


def render_selftest_mode(dfs: dict, progresses: dict) -> None:
    st.markdown("## Selbsttest")
    st.write("Wähle eine Anzahl an Fragen. Die Fragen werden zufällig aus beiden Datensätzen gezogen.")

    if not st.session_state.test_started:
        value = st.number_input(
            "Wie viele Fragen möchtest du testen?",
            min_value=1,
            max_value=len(build_test_pool(dfs)),
            value=int(st.session_state.test_num_questions),
            step=1,
        )
        st.session_state.test_num_questions = int(value)

        if st.button("Test beginnen", type="primary", use_container_width=True):
            start_test(dfs, int(value))
        return

    if st.session_state.test_finished:
        total = len(st.session_state.test_questions)
        correct = sum(1 for x in st.session_state.test_answers.values() if x["is_correct"])
        percent = (correct / total * 100) if total else 0.0

        apply_test_wrong_answers(progresses)

        st.success("Test abgeschlossen.")
        st.metric("Ergebnis", f"{percent:.1f}%")
        st.write(f"**Richtig:** {correct} von {total}")
        st.write(f"**Falsch:** {total - correct} von {total}")
        st.info("Nur die falsch beantworteten Fragen wurden nach Abschluss des Tests in Liste 0 ihres jeweiligen Fragenblocks verschoben.")

        if st.button("Neuen Test starten", use_container_width=True):
            reset_test_state()
            st.rerun()
        return

    current_index = st.session_state.test_current_index
    total_questions = len(st.session_state.test_questions)
    current_item = st.session_state.test_questions[current_index]
    dataset_key = current_item["dataset_key"]
    qid = current_item["question_id"]

    config = DATASETS[dataset_key]
    df = dfs[dataset_key]
    row = get_question_row(df, qid)
    q_type = question_type(row, config["correct_cols"])
    correct_idx = correct_indices(row, config["correct_cols"])
    correct_set = set(correct_idx)

    st.caption(f"Testfrage {current_index + 1} von {total_questions} · Quelle: {config['label']}")
    render_question_text(row, q_type)

    answer_key = f"test_{current_index}_{dataset_key}_{qid}"

    if not st.session_state.test_show_feedback:
        selected_indices = render_answer_inputs(
            prefix=answer_key,
            row=row,
            q_type=q_type,
            answer_cols=config["answer_cols"],
            option_keys=config["option_keys"],
            disabled=False,
        )

        if st.button("Antwort prüfen", type="primary", use_container_width=True, key=f"test_submit_{current_index}"):
            selected_set = set(selected_indices)
            is_correct = selected_set == correct_set

            st.session_state.test_answers[current_index] = {
                "dataset_key": dataset_key,
                "question_id": qid,
                "selected_indices": selected_indices,
                "correct_indices": correct_idx,
                "is_correct": is_correct,
            }
            st.session_state.test_show_feedback = True
            st.rerun()

    else:
        result = st.session_state.test_answers[current_index]

        if result["is_correct"]:
            st.success("Richtig.")
        else:
            st.error("Falsch.")

        selected_letters = letters_from_indices(result["selected_indices"], config["option_keys"])
        correct_letters_list = letters_from_indices(result["correct_indices"], config["option_keys"])

        st.write(f"**Deine Antwort:** {', '.join(selected_letters) if selected_letters else 'Keine Antwort ausgewählt'}")
        st.write(f"**Richtige Antwort(en):** {', '.join(correct_letters_list)}")

        st.write("**Richtige Antworttexte:**")
        for text in correct_answers_text(row, result["correct_indices"], config["answer_cols"], config["option_keys"]):
            st.write(f"- {text}")

        st.write("**Alle Antwortoptionen:**")
        for i in range(len(config["option_keys"])):
            marker = "✅" if i in correct_set else "❌"
            selected_marker = " ← deine Auswahl" if i in result["selected_indices"] else ""
            st.write(f"{marker} {config['option_keys'][i]}: {row[config['answer_cols'][i]]}{selected_marker}")

        button_label = "Test abschließen" if current_index == total_questions - 1 else "Nächste Testfrage"
        if st.button(button_label, type="primary", use_container_width=True, key=f"test_next_{current_index}"):
            clear_question_widget_state(answer_key, len(config["option_keys"]))
            st.session_state.test_show_feedback = False

            if current_index == total_questions - 1:
                st.session_state.test_finished = True
            else:
                st.session_state.test_current_index += 1

            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Flashcard Trainer", page_icon="🧠", layout="wide")
    st.title("🧠 Flashcard Trainer")

    initialize_session_state()

    try:
        dfs, progresses = load_all_data()
    except Exception as exc:
        st.error(str(exc))
        st.info("Bitte stelle sicher, dass set1-7.csv und set5.csv im selben Ordner wie app.py liegen.")
        st.stop()

    render_sidebar(dfs, progresses)

    if st.session_state.app_mode == "train4":
        render_training_mode("train4", dfs["train4"], progresses["train4"])
    elif st.session_state.app_mode == "train5":
        render_training_mode("train5", dfs["train5"], progresses["train5"])
    else:
        render_selftest_mode(dfs, progresses)


if __name__ == "__main__":
    main()
