from __future__ import annotations

import html
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mlqa.result import CheckStatus, ValidationResult
from mlqa.runner import run_validation_from_files


APP_TITLE = "ML QA Framework"
CHESHIRE_IMAGE_PATH = Path("assets/cheshire_loader.png")
MIN_CAT_SECONDS = 2.0


STATUS_LABELS = {
    CheckStatus.PASS: "ПРОЙДЕНО",
    CheckStatus.FAIL: "НЕ ПРОЙДЕНО",
    CheckStatus.WARN: "ПРЕДУПРЕЖДЕНИЕ",
    CheckStatus.SKIP: "ПРОПУЩЕНО",
}

STATUS_ICONS = {
    CheckStatus.PASS: "✓",
    CheckStatus.FAIL: "✕",
    CheckStatus.WARN: "!",
    CheckStatus.SKIP: "—",
}

STATUS_CLASSES = {
    CheckStatus.PASS: "pass",
    CheckStatus.FAIL: "fail",
    CheckStatus.WARN: "warn",
    CheckStatus.SKIP: "skip",
}


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🐈‍⬛",
        layout="centered",
        initial_sidebar_state="collapsed",
    )


def apply_styles() -> None:
    st.markdown(
        """
<style>
    :root {
        --bg: #09080d;
        --panel: #121018;
        --panel-soft: #0e0c12;
        --border: #34283f;

        --text: #f3f0f7;
        --muted: #aaa4b1;

        --purple: #8755d9;
        --purple-light: #b798ec;

        --pass: #4ad295;
        --pass-bg: rgba(74, 210, 149, 0.09);
        --pass-border: rgba(74, 210, 149, 0.55);

        --fail: #ff6579;
        --fail-bg: rgba(255, 101, 121, 0.09);
        --fail-border: rgba(255, 101, 121, 0.55);

        --warn: #f0bd58;
        --warn-bg: rgba(240, 189, 88, 0.09);
        --warn-border: rgba(240, 189, 88, 0.55);

        --skip: #d0aa61;
        --skip-bg: rgba(208, 170, 97, 0.06);
        --skip-border: rgba(208, 170, 97, 0.38);
    }

    .stApp {
        background:
            radial-gradient(
                circle at top center,
                rgba(111, 64, 166, 0.17),
                transparent 34rem
            ),
            var(--bg);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2.4rem;
        padding-bottom: 5rem;
    }

    h1 {
        color: var(--text) !important;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        margin-bottom: 0 !important;
    }

    h2,
    h3,
    h4 {
        color: var(--text) !important;
    }

    p,
    label,
    li {
        color: #d7d2dc;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .mlqa-author {
        text-align: center;
        color: var(--purple-light);
        font-size: 1.25rem;
        font-weight: 650;
        margin-top: 0.35rem;
        margin-bottom: 1.4rem;
    }

    .mlqa-intro {
        max-width: 720px;
        margin: 0 auto 2.4rem auto;
        text-align: center;
        color: var(--muted);
        line-height: 1.65;
    }

    .upload-number {
        color: var(--purple-light);
        font-weight: 800;
    }

    .optional-label {
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 700;
    }

    div[data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.85rem 1rem 0.45rem 1rem;
        margin-bottom: 1.7rem;
    }

    div[data-testid="stFileUploader"] section {
        background: var(--panel-soft);
        border-color: #493758;
        border-radius: 9px;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: var(--purple);
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        min-height: 3.25rem;
        border-radius: 9px;
        font-weight: 750;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #6840a6,
            #8755d9
        );
        border: 1px solid #9a6ee0;
        color: white;
    }

    div[data-testid="stButton"] > button[kind="secondary"] {
        background: var(--panel);
        border: 1px solid #4c3a5e;
        color: var(--text);
    }

    .summary-card {
        min-height: 100px;
        border-radius: 11px;
        border: 1px solid;
        padding: 1rem 0.5rem;
        text-align: center;
    }

    .summary-number {
        font-size: 1.8rem;
        line-height: 1;
        font-weight: 850;
    }

    .summary-label {
        margin-top: 0.55rem;
        font-size: 0.72rem;
        font-weight: 800;
    }

    .summary-card.pass {
        color: var(--pass);
        background: var(--pass-bg);
        border-color: var(--pass-border);
    }

    .summary-card.fail {
        color: var(--fail);
        background: var(--fail-bg);
        border-color: var(--fail-border);
    }

    .summary-card.warn {
        color: var(--warn);
        background: var(--warn-bg);
        border-color: var(--warn-border);
    }

    .summary-card.skip {
        color: var(--skip);
        background: var(--skip-bg);
        border-color: var(--skip-border);
    }

    .result-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;

        margin-top: 1rem;
        padding: 1.05rem 1.15rem;

        border: 1px solid;
        border-left-width: 5px;
        border-radius: 11px;
    }

    .result-title {
        color: var(--text);
        font-size: 1rem;
        font-weight: 750;
    }

    .result-status {
        white-space: nowrap;
        font-size: 0.82rem;
        font-weight: 850;
    }

    .result-card.pass {
        background: var(--pass-bg);
        border-color: var(--pass-border);
    }

    .result-card.pass .result-status {
        color: var(--pass);
    }

    .result-card.fail {
        background: var(--fail-bg);
        border-color: var(--fail-border);
    }

    .result-card.fail .result-status {
        color: var(--fail);
    }

    .result-card.warn {
        background: var(--warn-bg);
        border-color: var(--warn-border);
    }

    .result-card.warn .result-status {
        color: var(--warn);
    }

    .result-card.skip {
        background: var(--skip-bg);
        border-color: var(--skip-border);
    }

    .result-card.skip .result-status {
        color: var(--skip);
    }

    div[data-testid="stExpander"] {
        background: var(--panel-soft);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-top: 0.3rem;
        margin-bottom: 0.5rem;
    }

    .detail-title {
        color: var(--purple-light);
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: 0.3rem;
        margin-bottom: 0.8rem;
    }

    .fact-row {
        display: grid;
        grid-template-columns: minmax(220px, 0.9fr) 1.6fr;
        gap: 1rem;

        padding: 0.65rem 0;
        border-bottom: 1px solid rgba(80, 65, 92, 0.35);
    }

    .fact-name {
        color: var(--muted);
        font-weight: 600;
    }

    .fact-value {
        color: var(--text);
        overflow-wrap: anywhere;
    }

    .detail-description {
        color: #d7d2dc;
        line-height: 1.6;
        margin: 0.2rem 0 1rem 0;
    }

    .loader-title {
        text-align: center;
        color: var(--text);
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 1.2rem;
    }

    .loader-text {
        text-align: center;
        color: var(--muted);
        margin-top: 0.4rem;
    }

    hr {
        border-color: var(--border) !important;
    }

    @media (max-width: 700px) {
        h1 {
            font-size: 2.25rem !important;
        }

        .result-card {
            flex-direction: column;
            align-items: flex-start;
        }

        .fact-row {
            grid-template-columns: 1fr;
            gap: 0.25rem;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "screen": "upload",
        "validation_results": None,
        "validation_error": None,
        "technical_error": None,
        "temporary_directory": None,
        "pending_paths": None,
        "uploader_generation": 0,
        "loader_shown_at": None,
        "validation_started": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def remove_temporary_directory() -> None:
    directory = st.session_state.get(
        "temporary_directory"
    )

    if directory:
        shutil.rmtree(
            Path(directory),
            ignore_errors=True,
        )

    st.session_state.temporary_directory = None


def clear_framework() -> None:
    remove_temporary_directory()

    st.session_state.screen = "upload"
    st.session_state.validation_results = None
    st.session_state.validation_error = None
    st.session_state.technical_error = None
    st.session_state.pending_paths = None
    st.session_state.loader_shown_at = None
    st.session_state.validation_started = False
    st.session_state.uploader_generation += 1


def render_header(
    show_intro: bool = True,
) -> None:
    st.title(APP_TITLE)

    st.markdown(
        '<div class="mlqa-author">'
        'by Anna Gamgiya'
        '</div>',
        unsafe_allow_html=True,
    )

    if show_intro:
        st.markdown(
            """
            <div class="mlqa-intro">
                Проверка готовой обученной ML-модели
                с позиции QA-инженера:
                формальное требование, фактический результат
                и объективный статус проверки.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_upload_heading(
    number: int,
    title: str,
    required: bool,
) -> None:
    optional = ""

    if not required:
        optional = (
            ' <span class="optional-label">'
            '· НЕОБЯЗАТЕЛЬНО'
            '</span>'
        )

    st.markdown(
        f"""
        <h3>
            <span class="upload-number">{number}.</span>
            {html.escape(title)}
            {optional}
        </h3>
        """,
        unsafe_allow_html=True,
    )


def render_upload_screen() -> None:
    render_header()

    st.header("Что понадобится")

    st.write(
        "Для полного запуска подготовьте артефакт обученной модели, "
        "тестовый датасет и metadata с формальными требованиями. "
        "Target тестового датасета — по желанию."
    )

    st.divider()

    generation = st.session_state.uploader_generation

    render_upload_heading(
        1,
        "Модель",
        True,
    )

    st.write(
        "Готовая обученная ML-модель, которую нужно протестировать."
    )

    st.markdown(
        "**Поддерживаемый формат:** `.joblib`"
    )

    model_file = st.file_uploader(
        "Выбрать модель",
        type=["joblib"],
        key=f"model_{generation}",
    )

    render_upload_heading(
        2,
        "Тестовый датасет",
        True,
    )

    st.write(
        "Датасет с признаками, на которых будет тестироваться модель. "
        "Структура датасета проверяется по metadata "
        "и формальным данным самой модели."
    )

    st.markdown(
        "**Поддерживаемые форматы:** `.parquet`, `.csv`"
    )

    dataset_file = st.file_uploader(
        "Выбрать тестовый датасет",
        type=["parquet", "csv"],
        key=f"dataset_{generation}",
    )

    render_upload_heading(
        3,
        "Metadata",
        True,
    )

    st.write(
        "Машиночитаемый файл с формальными требованиями "
        "к проверяемой модели: обязательными признаками, target, "
        "документированными метриками и версиями библиотек."
    )

    st.markdown(
        "**Поддерживаемый формат:** `.json`"
    )

    metadata_file = st.file_uploader(
        "Выбрать metadata",
        type=["json"],
        key=f"metadata_{generation}",
    )

    render_upload_heading(
        4,
        "Target тестового датасета",
        False,
    )

    st.write(
        "Фактические значения target для строк тестового датасета. "
        "Используются для пересчёта документированных метрик "
        "и сравнения с их значениями из metadata."
    )

    st.caption(
        "Без target проверка воспроизводимости метрик будет SKIP."
    )

    st.markdown(
        "**Поддерживаемый формат:** `.csv`"
    )

    target_file = st.file_uploader(
        "Выбрать target",
        type=["csv"],
        key=f"target_{generation}",
    )

    st.divider()

    if st.button(
        "ЗАГРУЗИТЬ ДАННЫЕ",
        type="primary",
        use_container_width=True,
    ):
        prepare_run(
            model_file=model_file,
            dataset_file=dataset_file,
            metadata_file=metadata_file,
            target_file=target_file,
        )

    if st.button(
        "ОЧИСТИТЬ FRAMEWORK",
        type="secondary",
        use_container_width=True,
    ):
        clear_framework()
        st.rerun()

    render_error()


def validate_uploads(
    model_file: Any,
    dataset_file: Any,
    metadata_file: Any,
) -> list[str]:
    missing = []

    if model_file is None:
        missing.append("модель")

    if dataset_file is None:
        missing.append("тестовый датасет")

    if metadata_file is None:
        missing.append("metadata")

    return missing


def validate_metadata(
    metadata_file: Any,
) -> str | None:
    try:
        json.loads(
            metadata_file.getvalue().decode("utf-8")
        )

    except UnicodeDecodeError:
        return (
            "Metadata должна быть JSON-файлом "
            "в кодировке UTF-8."
        )

    except json.JSONDecodeError as exc:
        return (
            "Metadata содержит некорректный JSON. "
            f"Строка {exc.lineno}, столбец {exc.colno}."
        )

    return None


def save_uploaded_file(
    uploaded_file: Any,
    directory: Path,
) -> Path:
    path = directory / uploaded_file.name

    with path.open("wb") as handle:
        handle.write(
            uploaded_file.getbuffer()
        )

    return path


def prepare_run(
    *,
    model_file: Any,
    dataset_file: Any,
    metadata_file: Any,
    target_file: Any | None,
) -> None:
    missing = validate_uploads(
        model_file,
        dataset_file,
        metadata_file,
    )

    if missing:
        st.session_state.validation_error = (
            "Не выбраны обязательные файлы: "
            + ", ".join(missing)
            + "."
        )
        st.session_state.technical_error = None
        return

    metadata_error = validate_metadata(
        metadata_file
    )

    if metadata_error:
        st.session_state.validation_error = metadata_error
        st.session_state.technical_error = None
        return

    remove_temporary_directory()

    directory = Path(
        tempfile.mkdtemp(
            prefix="mlqa_"
        )
    )

    st.session_state.temporary_directory = str(
        directory
    )

    try:
        model_path = save_uploaded_file(
            model_file,
            directory,
        )

        dataset_path = save_uploaded_file(
            dataset_file,
            directory,
        )

        metadata_path = save_uploaded_file(
            metadata_file,
            directory,
        )

        target_path = None

        if target_file is not None:
            target_path = save_uploaded_file(
                target_file,
                directory,
            )

        st.session_state.pending_paths = {
            "model": str(model_path),
            "dataset": str(dataset_path),
            "metadata": str(metadata_path),
            "target": (
                str(target_path)
                if target_path is not None
                else None
            ),
        }

        st.session_state.validation_error = None
        st.session_state.technical_error = None

        # ВАЖНО:
        # новый запуск всегда начинает новый жизненный цикл loader.
        st.session_state.loader_shown_at = None
        st.session_state.validation_started = False

        st.session_state.screen = "running"

        st.rerun()

    except Exception as exc:
        st.session_state.validation_error = (
            "Не удалось подготовить файлы для проверки."
        )
        st.session_state.technical_error = (
            f"{type(exc).__name__}: {exc}"
        )


def render_cat() -> None:
    if CHESHIRE_IMAGE_PATH.exists():
        left, center, right = st.columns(
            [1.1, 1.25, 1.1]
        )

        with center:
            st.image(
                str(CHESHIRE_IMAGE_PATH),
                use_container_width=True,
            )
    else:
        st.error(
            "Не найден assets/cheshire_loader.png"
        )

    st.markdown(
        '<div class="loader-title">'
        'Выполняются QA-проверки'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="loader-text">'
        'Проверяется загруженный ML-пакет.'
        '</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=0.5)
def running_fragment() -> None:
    """
    Отдельный жизненный цикл loader.

    Первый проход:
        только показывает кота.

    Последующие автоматические проходы:
        ждут минимум 2 секунды.

    После истечения 2 секунд:
        один раз запускают validation.
    """

    render_cat()

    now = time.monotonic()

    # Первый проход.
    # Только запоминаем время появления loader
    # и НИЧЕГО больше не запускаем.
    if st.session_state.loader_shown_at is None:
        st.session_state.loader_shown_at = now
        return

    elapsed = (
        now
        - st.session_state.loader_shown_at
    )

    # Кот должен реально находиться на странице
    # не меньше двух секунд.
    if elapsed < MIN_CAT_SECONDS:
        return

    # Защита от повторного запуска validation
    # при очередном fragment rerun.
    if st.session_state.validation_started:
        return

    st.session_state.validation_started = True

    pending = st.session_state.pending_paths

    if not pending:
        st.session_state.validation_error = (
            "Не найдены подготовленные файлы для проверки."
        )
        st.session_state.technical_error = None
        st.session_state.screen = "error"
        st.rerun()

    try:
        results = run_validation_from_files(
            model_path=pending["model"],
            X_path=pending["dataset"],
            metadata_path=pending["metadata"],
            y_path=pending["target"],
        )

        st.session_state.validation_results = results
        st.session_state.validation_error = None
        st.session_state.technical_error = None
        st.session_state.pending_paths = None

        st.session_state.loader_shown_at = None
        st.session_state.validation_started = False

        st.session_state.screen = "results"

        st.rerun()

    except Exception as exc:
        st.session_state.validation_results = None

        st.session_state.validation_error = (
            "Не удалось выполнить проверку ML-пакета."
        )

        st.session_state.technical_error = (
            f"{type(exc).__name__}: {exc}"
        )

        st.session_state.pending_paths = None
        st.session_state.loader_shown_at = None
        st.session_state.validation_started = False

        st.session_state.screen = "error"

        st.rerun()


def render_running_screen() -> None:
    render_header(
        show_intro=False,
    )

    st.divider()

    running_fragment()


def render_summary(
    results: list[ValidationResult],
) -> None:
    counts = {
        CheckStatus.PASS: 0,
        CheckStatus.WARN: 0,
        CheckStatus.FAIL: 0,
        CheckStatus.SKIP: 0,
    }

    for result in results:
        counts[result.status] += 1

    order = [
        CheckStatus.PASS,
        CheckStatus.WARN,
        CheckStatus.FAIL,
        CheckStatus.SKIP,
    ]

    columns = st.columns(4)

    for column, status in zip(
        columns,
        order,
    ):
        with column:
            st.markdown(
                f"""
                <div class="summary-card {STATUS_CLASSES[status]}">
                    <div class="summary-number">
                        {STATUS_ICONS[status]} {counts[status]}
                    </div>
                    <div class="summary-label">
                        {STATUS_LABELS[status]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_result_header(
    result: ValidationResult,
    number: int,
) -> None:
    css_class = STATUS_CLASSES[
        result.status
    ]

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-title">
                {number}. {html.escape(result.check_name)}
            </div>
            <div class="result-status">
                {STATUS_ICONS[result.status]}
                {STATUS_LABELS[result.status]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fact(
    name: str,
    value: Any,
    status: CheckStatus | None = None,
) -> None:
    if value is None:
        value = "—"

    if isinstance(value, bool):
        value = "Да" if value else "Нет"

    if isinstance(value, (list, tuple, set)):
        value = "нет" if not value else ", ".join(str(item) for item in value)

    st.markdown(
        f"""
        <div class="fact-row">
            <div class="fact-name">{html.escape(str(name))}</div>
            <div class="fact-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_description(text: str) -> None:
    st.markdown(
        f'<div class="detail-description">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def evidence_of(
    assertion: Any,
) -> dict[str, Any]:
    if isinstance(
        assertion.evidence,
        dict,
    ):
        return assertion.evidence

    return {}


def render_issue(
    assertion: Any,
) -> None:
    if assertion.status == CheckStatus.PASS:
        return

    st.markdown(
        f"**{STATUS_ICONS[assertion.status]} "
        f"{STATUS_LABELS[assertion.status]}**"
    )

    if assertion.reason:
        st.write(
            assertion.reason
        )

    if assertion.evidence:
        if isinstance(
            assertion.evidence,
            (dict, list),
        ):
            st.code(
                json.dumps(
                    assertion.evidence,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                language=None,
            )
        else:
            st.code(
                str(assertion.evidence),
                language=None,
            )


def render_input_data_details(
    result: ValidationResult,
) -> None:
    render_description(
        "Проверяется соответствие структуры тестового датасета "
        "формальным требованиям metadata и модели."
    )

    assertions = result.assertions

    if len(assertions) >= 1:
        assertion = assertions[0]
        evidence = evidence_of(assertion)
        missing = evidence.get("отсутствующие_признаки", [])

        required_count = (
            len(assertion.expected)
            if isinstance(assertion.expected, (list, tuple))
            else None
        )

        if required_count is not None:
            present_count = max(required_count - len(missing), 0)
            required_value = f"{present_count} из {required_count}"
        else:
            required_value = "проверено"

        render_fact("Обязательные признаки", required_value)

        if missing:
            render_fact("Отсутствующие признаки", missing)

    if len(assertions) >= 2:
        assertion = assertions[1]
        duplicates = evidence_of(assertion).get("повторяющиеся_столбцы", [])
        render_fact(
            "Повторяющиеся столбцы",
            "нет" if not duplicates else duplicates,
        )

    if len(assertions) >= 3:
        assertion = assertions[2]
        evidence = evidence_of(assertion)
        missing_model = evidence.get("отсутствуют_в_модели", [])
        extra_model = evidence.get("лишние_в_модели", [])
        order_matches = evidence.get("порядок_совпадает")

        render_fact(
            "Структура признаков",
            "соответствует"
            if assertion.status == CheckStatus.PASS
            else "не соответствует",
        )

        if missing_model:
            render_fact("Отсутствуют в модели", missing_model)

        if extra_model:
            render_fact("Лишние в модели", extra_model)

        if order_matches is False:
            render_fact("Порядок признаков", "не совпадает")

    if not assertions and result.reason:
        render_fact("Причина", result.reason)


def render_prediction_execution_details(
    result: ValidationResult,
) -> None:
    render_description(
        "Проверяется способность готовой модели получить предсказания "
        "на загруженном тестовом датасете."
    )

    if not result.assertions:
        if result.reason:
            render_fact("Причина", result.reason)
        return

    assertion = result.assertions[0]
    evidence = evidence_of(assertion)
    exception = evidence.get("исключение")

    render_fact(
        "Техническая ошибка",
        "не возникла"
        if assertion.status == CheckStatus.PASS
        else "возникла",
    )

    if exception:
        render_fact("Ошибка", exception)

    if assertion.status != CheckStatus.PASS and assertion.reason:
        render_fact("Причина", assertion.reason)


def render_prediction_output_details(
    result: ValidationResult,
) -> None:
    render_description(
        "Проверяется структура полученных предсказаний и вероятностей "
        "относительно формального контракта модели."
    )

    for assertion in result.assertions:
        evidence = evidence_of(
            assertion
        )

        if (
            "количество_предсказаний"
            in evidence
        ):
            render_fact(
                "Количество предсказаний",
                (
                    f'{evidence.get("количество_предсказаний")} '
                    f'из '
                    f'{evidence.get("ожидаемое_количество_строк")}'
                ),
                assertion.status,
            )
            continue

        if (
            "недопустимые_предсказания"
            in evidence
        ):
            invalid = evidence.get(
                "недопустимые_предсказания",
                [],
            )

            render_fact(
                "Предсказанные классы",
                (
                    "соответствуют классам модели"
                    if not invalid
                    else f"недопустимые: {invalid}"
                ),
                assertion.status,
            )
            continue

        if (
            "количество_строк_вероятностей"
            in evidence
        ):
            render_fact(
                "Строки с вероятностями",
                (
                    f'{evidence.get("количество_строк_вероятностей")} '
                    f'из '
                    f'{evidence.get("ожидаемое_количество_строк")}'
                ),
                assertion.status,
            )
            continue

        if (
            "количество_столбцов_вероятностей"
            in evidence
        ):
            render_fact(
                "Столбцы вероятностей",
                (
                    f'{evidence.get("количество_столбцов_вероятностей")} '
                    f'из '
                    f'{evidence.get("ожидаемое_количество_столбцов")}'
                ),
                assertion.status,
            )
            continue

        if (
            "индексы_неконечных_значений"
            in evidence
        ):
            invalid = evidence.get(
                "индексы_неконечных_значений",
                [],
            )

            render_fact(
                "Неконечные вероятности",
                (
                    "нет"
                    if not invalid
                    else len(invalid)
                ),
                assertion.status,
            )
            continue

        if (
            "индексы_значений_вне_диапазона"
            in evidence
        ):
            invalid = evidence.get(
                "индексы_значений_вне_диапазона",
                [],
            )

            render_fact(
                "Вероятности вне диапазона [0, 1]",
                (
                    "нет"
                    if not invalid
                    else len(invalid)
                ),
                assertion.status,
            )
            continue

        if (
            "количество_строк_с_нарушением"
            in evidence
        ):
            invalid_count = evidence.get(
                "количество_строк_с_нарушением"
            )

            tolerance = evidence.get(
                "абсолютный_допуск"
            )

            render_fact(
                "Сумма вероятностей по строкам",
                (
                    "соответствует 1"
                    if invalid_count == 0
                    else (
                        "строк с нарушением: "
                        f"{invalid_count}"
                    )
                ),
                assertion.status,
            )

            if tolerance is not None:
                render_fact(
                    "Численный допуск",
                    tolerance,
                )

            continue

        render_issue(
            assertion
        )


def metric_name(
    assertion: Any,
) -> str:
    evidence = evidence_of(
        assertion
    )

    if evidence.get("метрика"):
        return str(
            evidence["метрика"]
        )

    requirement = str(
        assertion.requirement
    )

    prefix = (
        "Воспроизвести документированную метрику "
    )

    if requirement.startswith(
        prefix
    ):
        return requirement[
            len(prefix):
        ].rstrip(".")

    return requirement


def render_documented_metrics_details(
    result: ValidationResult,
) -> None:
    render_description(
        "Проверяется воспроизводимость метрик, сохранённых для модели "
        "в metadata, на загруженном тестовом датасете."
    )

    rows = []

    for assertion in result.assertions:
        evidence = evidence_of(
            assertion
        )

        actual = evidence.get(
            "получено_после_округления",
            assertion.actual,
        )

        rows.append(
            {
                "Метрика": metric_name(
                    assertion
                ),
                "Metadata": assertion.expected,
                "Пересчитано": (
                    actual
                    if actual is not None
                    else "—"
                ),
                "Статус": (
                    f"{STATUS_ICONS[assertion.status]} "
                    f"{assertion.status.value}"
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    skipped = [
        assertion
        for assertion in result.assertions
        if assertion.status == CheckStatus.SKIP
    ]

    if skipped:
        st.markdown(
            '<div class="detail-title">'
            'Почему некоторые метрики пропущены'
            '</div>',
            unsafe_allow_html=True,
        )

        for assertion in skipped:
            st.markdown(
                f"**{html.escape(metric_name(assertion))}:** "
                f"{html.escape(assertion.reason)}"
            )


def render_environment_details(
    result: ValidationResult,
) -> None:
    render_description(
        "Проверяется соответствие версий библиотек текущего окружения "
        "версиям, записанным для сохранённой модели."
    )

    rows = []

    for assertion in result.assertions:
        evidence = evidence_of(
            assertion
        )

        library = evidence.get(
            "библиотека",
            assertion.requirement,
        )

        expected = evidence.get(
            "ожидаемая_версия",
            assertion.expected,
        )

        actual = evidence.get(
            "фактическая_версия",
            assertion.actual,
        )

        rows.append(
            {
                "Библиотека": library,
                "Metadata": expected,
                "Окружение": actual,
                "Статус": (
                    f"{STATUS_ICONS[assertion.status]} "
                    f"{assertion.status.value}"
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    if result.status == CheckStatus.WARN:
        st.caption(
            "Несовпадение версии само по себе "
            "не означает неработоспособность модели."
        )


def _missing_features_from_result(result: ValidationResult) -> list[str]:
    for assertion in result.assertions:
        evidence = evidence_of(assertion)
        missing = evidence.get("отсутствующие_признаки")
        if isinstance(missing, (list, tuple, set)) and missing:
            return [str(item) for item in missing]

        message = evidence.get("сообщение_исключения")
        if not message:
            continue

        text = str(message)
        marker = "columns are missing:"
        if marker not in text:
            continue

        raw = text.split(marker, 1)[1].strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]

        names = [
            item.strip().strip("'\"")
            for item in raw.split(",")
            if item.strip().strip("'\"")
        ]
        if names:
            return names

    return []


def _reason_for_ui(result: ValidationResult) -> str:
    missing = _missing_features_from_result(result)
    if len(missing) == 1:
        return f"В тестовом датасете отсутствует признак {missing[0]}."
    if len(missing) > 1:
        return "В тестовом датасете отсутствуют признаки: " + ", ".join(missing) + "."

    reasons = []
    for assertion in result.assertions:
        if assertion.status != CheckStatus.PASS and assertion.reason:
            reason = str(assertion.reason).strip()
            if reason and reason not in reasons:
                reasons.append(reason)

    if reasons:
        return " ".join(reasons)

    return result.reason or "Причина не указана."


def render_technical_details(
    result: ValidationResult,
) -> None:
    render_fact("Источник требования", result.requirement_source)

    if result.status == CheckStatus.PASS:
        return

    render_fact("Тест", STATUS_LABELS[result.status].lower())
    render_fact("Причина", _reason_for_ui(result))

def render_result(
    result: ValidationResult,
    number: int,
) -> None:
    render_result_header(
        result,
        number,
    )

    with st.expander(
        "Технические детали",
        expanded=False,
    ):
        render_technical_details(
            result
        )


def render_results_screen() -> None:
    render_header(
        show_intro=False,
    )

    results = (
        st.session_state.validation_results
    )

    if not results:
        st.error(
            "Результаты проверки отсутствуют."
        )
        return

    st.header(
        "Результаты QA-проверки"
    )

    render_summary(
        results
    )

    st.divider()

    for number, result in enumerate(
        results,
        start=1,
    ):
        render_result(
            result,
            number,
        )

    st.divider()

    if st.button(
        "НОВАЯ ПРОВЕРКА",
        type="primary",
        use_container_width=True,
    ):
        clear_framework()
        st.rerun()

    if st.button(
        "ОЧИСТИТЬ FRAMEWORK",
        type="secondary",
        use_container_width=True,
    ):
        clear_framework()
        st.rerun()


def render_error() -> None:
    error = (
        st.session_state.validation_error
    )

    if not error:
        return

    st.error(
        error
    )

    technical_error = (
        st.session_state.technical_error
    )

    if technical_error:
        with st.expander(
            "Технические подробности",
            expanded=False,
        ):
            st.code(
                technical_error,
                language=None,
            )


def render_error_screen() -> None:
    render_header(
        show_intro=False,
    )

    st.header(
        "Проверка не выполнена"
    )

    render_error()

    if st.button(
        "ВЕРНУТЬСЯ К ЗАГРУЗКЕ",
        type="primary",
        use_container_width=True,
    ):
        clear_framework()
        st.rerun()


def main() -> None:
    configure_page()
    apply_styles()
    initialize_state()

    screen = (
        st.session_state.screen
    )

    if screen == "upload":
        render_upload_screen()
        return

    if screen == "running":
        render_running_screen()
        return

    if screen == "results":
        render_results_screen()
        return

    if screen == "error":
        render_error_screen()
        return

    clear_framework()
    st.rerun()


if __name__ == "__main__":
    main()