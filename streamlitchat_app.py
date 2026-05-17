from __future__ import annotations

import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import psycopg2
import requests
import streamlit as st


warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)

DB_CONFIG = {
    "host": "dataiesb.iesbtech.com.br",
    "port": 5432,
    "dbname": "2312120014_Carlos",
    "user": "2312120014_Carlos",
    "password": "2312120014_Carlos",
    "connect_timeout": 20,
}

FACT_TABLE = "public.sih_sus_aih_spabr_clean"
SUBGROUP_TABLE = "public.sih_subgrupo_procedimento"
LONG_VIEW = "public.vw_sih_sus_aih_spabr_long"
GEMINI_MODEL = "gemini-2.5-flash"

CONTENT_LABELS = {
    "qtd_aprovada": "Quantidade aprovada",
    "valor_aprovado": "Valor aprovado",
}

MONTHS = {
    "jan": 1,
    "janeiro": 1,
    "fev": 2,
    "fevereiro": 2,
    "mar": 3,
    "marco": 3,
    "abr": 4,
    "abril": 4,
    "mai": 5,
    "maio": 5,
    "jun": 6,
    "junho": 6,
    "jul": 7,
    "julho": 7,
    "ago": 8,
    "agosto": 8,
    "set": 9,
    "setembro": 9,
    "out": 10,
    "outubro": 10,
    "nov": 11,
    "novembro": 11,
    "dez": 12,
    "dezembro": 12,
}

UFS = {
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
}


def get_secret(name: str, default: str | None = None) -> str | None:
    env_value = os.environ.get(name)
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return default


def get_db_config() -> dict[str, Any]:
    return {
        "host": get_secret("POSTGRES_HOST", DB_CONFIG["host"]),
        "port": int(get_secret("POSTGRES_PORT", str(DB_CONFIG["port"])) or 5432),
        "dbname": get_secret("POSTGRES_DB", DB_CONFIG["dbname"]),
        "user": get_secret("POSTGRES_USER", DB_CONFIG["user"]),
        "password": get_secret("POSTGRES_PASSWORD", DB_CONFIG["password"]),
        "connect_timeout": 20,
    }


@dataclass
class Filters:
    content: str = "qtd_aprovada"
    period: date | None = None
    year: int | None = None
    uf: str | None = None
    municipality: str | None = None


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    return value.lower()


def format_number(value: Any, content: str) -> str:
    if value is None:
        return "0"
    number = float(value)
    if content == "valor_aprovado":
        return f"R$ {number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{number:,.0f}".replace(",", ".")


def run_query(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    conn = psycopg2.connect(**get_db_config())
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


@st.cache_data(ttl=600)
def cached_query(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return run_query(sql, params)


def get_gemini_api_key() -> str | None:
    session_key = st.session_state.get("gemini_api_key") if hasattr(st, "session_state") else None
    if session_key:
        return session_key.strip()

    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(key_name)
        if value:
            return value.strip()

    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_periods() -> pd.DataFrame:
    return cached_query(
        f"""
        select distinct periodo, periodo_rotulo
        from {FACT_TABLE}
        order by periodo
        """
    )


@st.cache_data(ttl=1800)
def get_ufs() -> list[str]:
    frame = cached_query(
        f"""
        select distinct municipio_uf
        from {FACT_TABLE}
        where municipio_uf is not null
        order by municipio_uf
        """
    )
    return frame["municipio_uf"].tolist()


@st.cache_data(ttl=1800)
def get_subgroups() -> pd.DataFrame:
    return cached_query(
        f"""
        select subgrupo_coluna, subgrupo_codigo, subgrupo_nome
        from {SUBGROUP_TABLE}
        order by subgrupo_coluna
        """
    )


def build_where(filters: Filters) -> tuple[str, list[Any]]:
    clauses = ["conteudo = %s"]
    params: list[Any] = [filters.content]

    if filters.period is not None:
        clauses.append("periodo = %s")
        params.append(filters.period)
    elif filters.year is not None:
        clauses.append("ano = %s")
        params.append(filters.year)

    if filters.uf:
        clauses.append("municipio_uf = %s")
        params.append(filters.uf)

    if filters.municipality:
        clauses.append("upper(municipio_nome) like upper(%s)")
        params.append(f"%{filters.municipality}%")

    return " and ".join(clauses), params


def sql_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, date):
        return f"'{value.isoformat()}'"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def build_display_where(filters: Filters, alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    clauses = [f"{prefix}conteudo = {sql_literal(filters.content)}"]

    if filters.period is not None:
        clauses.append(f"{prefix}periodo = {sql_literal(filters.period)}")
    elif filters.year is not None:
        clauses.append(f"{prefix}ano = {filters.year}")

    if filters.uf:
        clauses.append(f"{prefix}municipio_uf = {sql_literal(filters.uf)}")

    if filters.municipality:
        clauses.append(f"upper({prefix}municipio_nome) like upper({sql_literal('%' + filters.municipality + '%')})")

    return "\n  and ".join(clauses)


def sql_for_base_info() -> str:
    return f"""select
    count(*) as linhas,
    count(distinct periodo) as periodos,
    min(periodo) as primeiro_periodo,
    max(periodo) as ultimo_periodo,
    count(distinct municipio_raw) as municipios
from {FACT_TABLE};"""


def sql_for_metric_summary(filters: Filters) -> str:
    return f"""select
    count(*) as linhas,
    count(distinct municipio_raw) as municipios,
    sum(total_linha) as total,
    avg(total_linha) as media,
    max(total_linha) as maior
from {FACT_TABLE}
where {build_display_where(filters)};"""


def sql_for_timeline(filters: Filters) -> str:
    query_filters = Filters(content=filters.content, uf=filters.uf, municipality=filters.municipality)
    return f"""select
    periodo,
    periodo_rotulo,
    sum(total_linha) as total
from {FACT_TABLE}
where {build_display_where(query_filters)}
group by periodo, periodo_rotulo
order by periodo;"""


def sql_for_top_municipalities(filters: Filters, limit: int) -> str:
    return f"""select
    municipio_nome,
    municipio_uf,
    sum(total_linha) as total
from {FACT_TABLE}
where {build_display_where(filters)}
group by municipio_nome, municipio_uf
order by total desc
limit {limit};"""


def sql_for_top_ufs(filters: Filters, limit: int) -> str:
    return f"""select
    municipio_uf,
    sum(total_linha) as total
from {FACT_TABLE}
where {build_display_where(filters)}
  and municipio_uf is not null
group by municipio_uf
order by total desc
limit {limit};"""


def sql_for_top_subgroups(filters: Filters, limit: int) -> str:
    return f"""select
    subgrupo_codigo,
    subgrupo_nome,
    sum(valor) as total
from public.vw_sih_sus_aih_spabr_long
where {build_display_where(filters)}
group by subgrupo_codigo, subgrupo_nome
order by total desc
limit {limit};"""


def get_schema_prompt() -> str:
    return f"""
Voce e um agente gerador de SQL para PostgreSQL.
Gere apenas consultas SELECT seguras sobre a base SIH/SUS.

Tabelas disponiveis:
1. {FACT_TABLE}
   - periodo DATE
   - ano SMALLINT
   - mes SMALLINT
   - periodo_rotulo TEXT
   - conteudo TEXT: 'qtd_aprovada' ou 'valor_aprovado'
   - municipio_codigo INTEGER
   - municipio_nome TEXT
   - municipio_uf CHAR(2)
   - total_linha NUMERIC

2. {LONG_VIEW}
   - periodo DATE
   - ano SMALLINT
   - mes SMALLINT
   - periodo_rotulo TEXT
   - conteudo TEXT
   - municipio_codigo INTEGER
   - municipio_nome TEXT
   - municipio_uf CHAR(2)
   - subgrupo_codigo TEXT
   - subgrupo_nome TEXT
   - valor NUMERIC

Regras obrigatorias:
- Responda somente em JSON valido.
- Use somente SELECT.
- Nunca use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, COPY ou comandos administrativos.
- Para totais por municipio/UF/periodo use {FACT_TABLE} e total_linha.
- Para consultas por subgrupo/procedimento use {LONG_VIEW} e valor.
- Para "quantidade aprovada", use conteudo = 'qtd_aprovada'.
- Para "valor aprovado", use conteudo = 'valor_aprovado'.
- Datas mensais devem ser o primeiro dia do mes. Exemplo: janeiro de 2026 = DATE '2026-01-01'.
- Quando houver ranking ou maior/menor, use order by e limit.
- Se a pergunta for ampla, use limit 10.

Formato JSON:
{{
  "sql": "select ...",
  "chart_type": "bar|line|none",
  "content_key": "qtd_aprovada|valor_aprovado",
  "explanation": "frase curta em portugues explicando a consulta"
}}
""".strip()


def extract_json_from_text(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean).strip()
        clean = re.sub(r"```$", "", clean).strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end >= start:
        clean = clean[start : end + 1]
    return json.loads(clean)


def validate_generated_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    compact = re.sub(r"\s+", " ", sql).strip()
    lowered = compact.lower()

    if not lowered.startswith("select "):
        raise ValueError("A consulta gerada nao e SELECT.")

    forbidden = [
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " create ",
        " truncate ",
        " copy ",
        " grant ",
        " revoke ",
        " execute ",
        " call ",
        " do ",
        " set ",
    ]
    padded = f" {lowered} "
    if any(token in padded for token in forbidden):
        raise ValueError("A consulta gerada contem comando nao permitido.")

    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("A consulta gerada contem separadores ou comentarios nao permitidos.")

    allowed_sources = [FACT_TABLE, LONG_VIEW, SUBGROUP_TABLE]
    if not any(source.lower() in lowered for source in allowed_sources):
        raise ValueError("A consulta gerada nao usa uma tabela permitida.")

    if " limit " not in lowered and not any(term in lowered for term in ["sum(", "count(", "avg(", "max(", "min("]):
        sql = f"{sql}\nlimit 30"

    return sql + ";"


def call_gemini_for_sql(question: str) -> dict[str, Any]:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nao configurada.")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    prompt = f"{get_schema_prompt()}\n\nPergunta do usuario:\n{question}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }
    response = requests.post(
        endpoint,
        params={"key": api_key},
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = extract_json_from_text(text)
    parsed["sql"] = validate_generated_sql(parsed["sql"])
    parsed["chart_type"] = parsed.get("chart_type") or "none"
    parsed["content_key"] = parsed.get("content_key") or detect_content(question)
    return parsed


def summarize_llm_result(question: str, frame: pd.DataFrame, parsed: dict[str, Any]) -> str:
    content_key = parsed.get("content_key", detect_content(question))
    if frame.empty:
        return "O Gemini gerou a consulta, mas nao encontrei registros para esses filtros."

    first = frame.iloc[0]
    if "municipio_nome" in frame.columns and "total" in frame.columns:
        uf = f" ({first['municipio_uf']})" if "municipio_uf" in frame.columns else ""
        return f"Segundo a consulta gerada pelo Gemini, o principal municipio foi {first['municipio_nome']}{uf}, com {format_number(first['total'], content_key)}."

    if "municipio_uf" in frame.columns and "total" in frame.columns:
        return f"Segundo a consulta gerada pelo Gemini, a principal UF foi {first['municipio_uf']}, com {format_number(first['total'], content_key)}."

    if "subgrupo_nome" in frame.columns and "total" in frame.columns:
        return f"Segundo a consulta gerada pelo Gemini, o principal subgrupo foi {first['subgrupo_nome']}, com {format_number(first['total'], content_key)}."

    if "total" in frame.columns and len(frame) == 1:
        return f"Segundo a consulta gerada pelo Gemini, o total encontrado foi {format_number(first['total'], content_key)}."

    explanation = parsed.get("explanation") or "Consulta gerada pelo Gemini 2.5 Flash."
    return f"{explanation} Retornei {len(frame)} linha(s) para analise."


def metric_summary(filters: Filters) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    return cached_query(
        f"""
        select
            count(*) as linhas,
            count(distinct municipio_raw) as municipios,
            sum(total_linha) as total,
            avg(total_linha) as media,
            max(total_linha) as maior
        from {FACT_TABLE}
        where {where_sql}
        """,
        tuple(params),
    )


def timeline(filters: Filters) -> pd.DataFrame:
    query_filters = Filters(content=filters.content, uf=filters.uf, municipality=filters.municipality)
    where_sql, params = build_where(query_filters)
    return cached_query(
        f"""
        select periodo, periodo_rotulo, sum(total_linha) as total
        from {FACT_TABLE}
        where {where_sql}
        group by periodo, periodo_rotulo
        order by periodo
        """,
        tuple(params),
    )


def top_municipalities(filters: Filters, limit: int = 15) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    return cached_query(
        f"""
        select
            municipio_nome,
            municipio_uf,
            sum(total_linha) as total
        from {FACT_TABLE}
        where {where_sql}
        group by municipio_nome, municipio_uf
        order by total desc
        limit %s
        """,
        tuple(params + [limit]),
    )


def top_ufs(filters: Filters) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    return cached_query(
        f"""
        select municipio_uf, sum(total_linha) as total
        from {FACT_TABLE}
        where {where_sql}
          and municipio_uf is not null
        group by municipio_uf
        order by total desc
        """,
        tuple(params),
    )


def top_subgroups(filters: Filters, limit: int = 15) -> pd.DataFrame:
    subgroups = get_subgroups()
    values_sql = ",\n".join(
        f"('{row.subgrupo_coluna}', t.{row.subgrupo_coluna})"
        for row in subgroups.itertuples(index=False)
    )
    where_sql, params = build_where(filters)
    return cached_query(
        f"""
        select
            d.subgrupo_codigo,
            d.subgrupo_nome,
            sum(v.valor) as total
        from {FACT_TABLE} t
        cross join lateral (
            values
            {values_sql}
        ) as v(subgrupo_coluna, valor)
        join {SUBGROUP_TABLE} d
            on d.subgrupo_coluna = v.subgrupo_coluna
        where {where_sql}
        group by d.subgrupo_codigo, d.subgrupo_nome
        order by total desc
        limit %s
        """,
        tuple(params + [limit]),
    )


def data_sample(filters: Filters, limit: int = 200) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    return cached_query(
        f"""
        select
            periodo_rotulo,
            conteudo,
            municipio_codigo,
            municipio_nome,
            municipio_uf,
            total_linha
        from {FACT_TABLE}
        where {where_sql}
        order by periodo, municipio_uf, municipio_nome
        limit %s
        """,
        tuple(params + [limit]),
    )


def detect_content(question: str) -> str:
    text = normalize_text(question)
    if "valor" in text or "r$" in text or "dinheiro" in text:
        return "valor_aprovado"
    return "qtd_aprovada"


def detect_period(question: str) -> tuple[date | None, int | None]:
    text = normalize_text(question)

    iso_match = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", text)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        return date(year, month, 1), None

    month_year_match = re.search(
        r"\b("
        + "|".join(MONTHS)
        + r")\s*(?:/|de|\s)\s*(20\d{2})\b",
        text,
    )
    if month_year_match:
        month = MONTHS[month_year_match.group(1)]
        year = int(month_year_match.group(2))
        return date(year, month, 1), None

    year_match = re.search(r"\b(2024|2025|2026)\b", text)
    if year_match:
        return None, int(year_match.group(1))

    return None, None


def detect_uf(question: str) -> str | None:
    tokens = set(re.findall(r"\b[A-Z]{2}\b", question.upper()))
    matches = sorted(tokens & UFS)
    return matches[0] if matches else None


def detect_limit(question: str, default: int = 10) -> int:
    match = re.search(r"\btop\s*(\d{1,2})\b|\b(\d{1,2})\s+maior", normalize_text(question))
    if not match:
        return default
    value = int(next(group for group in match.groups() if group))
    return min(max(value, 1), 30)


def detect_municipality(question: str) -> str | None:
    text = normalize_text(question)
    patterns = [
        r"municipio de ([a-z\s]+)",
        r"cidade de ([a-z\s]+)",
    ]
    stop_words = {
        "janeiro",
        "fevereiro",
        "marco",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    }
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            candidate = re.split(r"\b(20\d{2}|valor|quantidade|qtd|aprovada|aprovado|total)\b", candidate)[0].strip()
            if candidate and candidate not in stop_words and len(candidate) >= 3:
                return candidate
    return None


def describe_filters(filters: Filters) -> str:
    parts = [CONTENT_LABELS[filters.content]]
    if filters.period:
        parts.append(filters.period.strftime("%m/%Y"))
    elif filters.year:
        parts.append(str(filters.year))
    else:
        parts.append("todos os periodos")
    if filters.uf:
        parts.append(filters.uf)
    if filters.municipality:
        parts.append(f"municipio contendo '{filters.municipality}'")
    return " | ".join(parts)


def answer_question_rules(question: str) -> tuple[str, pd.DataFrame | None, str | None, str | None, str]:
    text = normalize_text(question)
    period, year = detect_period(question)
    filters = Filters(
        content=detect_content(question),
        period=period,
        year=year,
        uf=detect_uf(question),
        municipality=detect_municipality(question),
    )
    limit = detect_limit(question)

    if any(term in text for term in ["periodos", "base", "registros", "linhas"]):
        frame = cached_query(
            f"""
            select
                count(*) as linhas,
                count(distinct periodo) as periodos,
                min(periodo) as primeiro_periodo,
                max(periodo) as ultimo_periodo,
                count(distinct municipio_raw) as municipios
            from {FACT_TABLE}
            """
        )
        row = frame.iloc[0]
        return (
            f"A base tem {int(row.linhas):,} linhas, {int(row.periodos)} periodos e "
            f"{int(row.municipios):,} municipios/linhas municipais.".replace(",", "."),
            frame,
            None,
            sql_for_base_info(),
            "Fallback por regras",
        )

    if "subgrupo" in text or "procedimento" in text:
        frame = top_subgroups(filters, limit=limit)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = (
            f"O maior subgrupo para {describe_filters(filters)} foi "
            f"{frame.iloc[0]['subgrupo_nome']} com {format_number(total, filters.content)}."
            if not frame.empty
            else "Nao encontrei resultado para esses filtros."
        )
        return message, frame, "bar", sql_for_top_subgroups(filters, limit), "Fallback por regras"

    if any(term in text for term in ["estado", "uf", "ufs"]):
        frame = top_ufs(filters)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = (
            f"A UF com maior total para {describe_filters(filters)} foi "
            f"{frame.iloc[0]['municipio_uf']} com {format_number(total, filters.content)}."
            if not frame.empty
            else "Nao encontrei resultado para esses filtros."
        )
        return message, frame.head(limit), "bar", sql_for_top_ufs(filters, limit), "Fallback por regras"

    if any(term in text for term in ["evolucao", "mensal", "por mes", "serie"]):
        frame = timeline(filters)
        return f"Evolucao mensal para {describe_filters(filters)}.", frame, "line", sql_for_timeline(filters), "Fallback por regras"

    if any(term in text for term in ["maior", "ranking", "top", "municipios", "cidades"]):
        frame = top_municipalities(filters, limit=limit)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = (
            f"O maior municipio para {describe_filters(filters)} foi "
            f"{frame.iloc[0]['municipio_nome']} ({frame.iloc[0]['municipio_uf']}) "
            f"com {format_number(total, filters.content)}."
            if not frame.empty
            else "Nao encontrei resultado para esses filtros."
        )
        return message, frame, "bar", sql_for_top_municipalities(filters, limit), "Fallback por regras"

    if "total" in text or "soma" in text:
        frame = metric_summary(filters)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = f"O total para {describe_filters(filters)} e {format_number(total, filters.content)}."
        return message, frame, None, sql_for_metric_summary(filters), "Fallback por regras"

    if filters.municipality:
        frame = top_municipalities(filters, limit=limit)
        return f"Resultado para {describe_filters(filters)}.", frame, "bar", sql_for_top_municipalities(filters, limit), "Fallback por regras"

    return (
        "Posso responder totais, maiores municipios, ranking por UF, ranking por subgrupo "
        "e evolucao mensal. Exemplo: total de quantidade aprovada em janeiro de 2026.",
        None,
        None,
        None,
        "Fallback por regras",
    )


def answer_question(question: str) -> tuple[str, pd.DataFrame | None, str | None, str | None, str]:
    if get_gemini_api_key():
        try:
            parsed = call_gemini_for_sql(question)
            frame = cached_query(parsed["sql"])
            message = summarize_llm_result(question, frame, parsed)
            return (
                message,
                frame,
                parsed.get("chart_type"),
                parsed["sql"],
                f"Gemini 2.5 Flash ({GEMINI_MODEL})",
            )
        except Exception as exc:
            fallback_answer, frame, chart_type, sql_query, _ = answer_question_rules(question)
            message = (
                f"Nao consegui usar o Gemini nesta pergunta, entao respondi pelo fallback por regras. "
                f"{fallback_answer}"
            )
            return message, frame, chart_type, sql_query, f"Fallback por regras - Gemini indisponivel: {exc}"

    return answer_question_rules(question)


def draw_result_chart(frame: pd.DataFrame, chart_type: str | None, content: str) -> None:
    if frame is None or frame.empty or chart_type is None:
        return

    y_label = CONTENT_LABELS[content]
    if chart_type == "line" and "periodo_rotulo" in frame.columns:
        fig = px.line(frame, x="periodo_rotulo", y="total", markers=True, labels={"total": y_label})
        st.plotly_chart(fig, width="stretch")
    elif chart_type == "bar":
        label_column = next(
            (
                column
                for column in ["municipio_nome", "municipio_uf", "subgrupo_nome"]
                if column in frame.columns
            ),
            frame.columns[0],
        )
        fig = px.bar(
            frame.sort_values("total"),
            x="total",
            y=label_column,
            orientation="h",
            labels={"total": y_label, label_column: ""},
        )
        st.plotly_chart(fig, width="stretch")


def render_sql_query(sql_query: str | None) -> None:
    if not sql_query:
        return
    with st.expander("Consulta SQL usada"):
        st.code(sql_query, language="sql")


def render_model_used(model_used: str | None) -> None:
    if not model_used:
        return
    st.caption(f"Agente gerador da query: {model_used}")


def render_dashboard() -> None:
    periods = get_periods()
    ufs = get_ufs()

    st.sidebar.header("Filtros")
    content_label = st.sidebar.selectbox("Conteudo", list(CONTENT_LABELS.values()))
    content = next(key for key, label in CONTENT_LABELS.items() if label == content_label)

    period_labels = ["Todos"] + periods["periodo_rotulo"].tolist()
    selected_period_label = st.sidebar.selectbox("Periodo", period_labels, index=len(period_labels) - 1)
    selected_period = None
    if selected_period_label != "Todos":
        selected_period = periods.loc[periods["periodo_rotulo"] == selected_period_label, "periodo"].iloc[0]

    selected_uf = st.sidebar.selectbox("UF", ["Todas"] + ufs)
    filters = Filters(
        content=content,
        period=selected_period,
        uf=None if selected_uf == "Todas" else selected_uf,
    )

    summary = metric_summary(filters).iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", format_number(summary["total"], content))
    col2.metric("Municipios", f"{int(summary['municipios']):,}".replace(",", "."))
    col3.metric("Media", format_number(summary["media"], content))
    col4.metric("Maior", format_number(summary["maior"], content))

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Evolucao mensal")
        frame = timeline(filters)
        fig = px.line(frame, x="periodo_rotulo", y="total", markers=True, labels={"total": content_label})
        st.plotly_chart(fig, width="stretch")

        st.subheader("Dados armazenados")
        st.dataframe(data_sample(filters), width="stretch", hide_index=True)

    with right:
        st.subheader("Top municipios")
        muni = top_municipalities(filters, limit=12)
        fig = px.bar(
            muni.sort_values("total"),
            x="total",
            y="municipio_nome",
            color="municipio_uf",
            orientation="h",
            labels={"total": content_label, "municipio_nome": ""},
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Top subgrupos")
        subgroup = top_subgroups(filters, limit=12)
        fig = px.bar(
            subgroup.sort_values("total"),
            x="total",
            y="subgrupo_nome",
            orientation="h",
            labels={"total": content_label, "subgrupo_nome": ""},
        )
        st.plotly_chart(fig, width="stretch")


def render_chat() -> None:
    st.markdown(
        """
        <section class="chat-hero">
            <div>
                <p class="eyebrow">SIH/SUS DATASUS</p>
                <h1>Chat da Producao Hospitalar</h1>
                <p class="hero-copy">Consulte a base carregada no PostgreSQL usando perguntas em linguagem natural.</p>
            </div>
            <div class="status-card">
                <span></span>
                Banco conectado
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    model_status = (
        f"Ativo: Gemini 2.5 Flash ({GEMINI_MODEL})"
        if get_gemini_api_key()
        else "Preparado para Gemini 2.5 Flash; usando fallback por regras ate configurar a chave"
    )
    st.markdown(
        f"""
        <div class="agent-card">
            <strong>Agente gerador de SQL</strong>
            <span>{model_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Configurar chave do Gemini"):
        st.text_input(
            "GEMINI_API_KEY",
            type="password",
            key="gemini_api_key",
            help="Cole aqui a chave do Google AI Studio para ativar o Gemini 2.5 Flash nesta sessao.",
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Oi, Carlos. Pode perguntar sobre quantidade aprovada, valor aprovado, municipios, UFs, periodos ou subgrupos.",
            }
        ]

    st.markdown('<div class="quick-prompts">', unsafe_allow_html=True)
    cols = st.columns(4)
    prompt_options = [
        "Total de quantidade aprovada em janeiro de 2026",
        "Maior valor aprovado em janeiro de 2026",
        "Top 5 estados por valor aprovado em 2025",
        "Top 5 subgrupos em SP em jan/2026",
    ]
    selected_prompt = None
    for col, option in zip(cols, prompt_options):
        with col:
            if st.button(option, use_container_width=True):
                selected_prompt = option
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            frame = message.get("frame")
            chart_type = message.get("chart_type")
            content = message.get("content_key", "qtd_aprovada")
            sql_query = message.get("sql_query")
            model_used = message.get("model_used")
            render_model_used(model_used)
            render_sql_query(sql_query)
            if isinstance(frame, pd.DataFrame):
                draw_result_chart(frame, chart_type, content)
                st.dataframe(frame, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Digite sua pergunta")
    prompt = selected_prompt or prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        answer, frame, chart_type, sql_query, model_used = answer_question(prompt)
        content_key = detect_content(prompt)
        response = {
            "role": "assistant",
            "content": answer,
            "frame": frame,
            "chart_type": chart_type,
            "content_key": content_key,
            "sql_query": sql_query,
            "model_used": model_used,
        }
        st.session_state.messages.append(response)
        with st.chat_message("assistant"):
            st.write(answer)
            render_model_used(model_used)
            render_sql_query(sql_query)
            if isinstance(frame, pd.DataFrame):
                draw_result_chart(frame, chart_type, content_key)
                st.dataframe(frame, width="stretch", hide_index=True)


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #16201f;
            --muted: #5c6764;
            --panel: rgba(255, 255, 255, 0.86);
            --line: rgba(22, 32, 31, 0.12);
            --teal: #0f766e;
            --mint: #d8f3ea;
            --coral: #f9735b;
            --gold: #f4c95d;
        }

        .stApp {
            background:
                radial-gradient(circle at 14% 12%, rgba(15, 118, 110, 0.16), transparent 30%),
                radial-gradient(circle at 86% 8%, rgba(249, 115, 91, 0.14), transparent 28%),
                linear-gradient(135deg, #f7faf8 0%, #edf5f1 48%, #f8f2ee 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 980px;
            padding-top: 2.1rem;
            padding-bottom: 7rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .chat-hero {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 18px 50px rgba(22, 32, 31, 0.10);
            backdrop-filter: blur(18px);
        }

        .eyebrow {
            margin: 0 0 0.35rem 0;
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
        }

        .chat-hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.25rem);
            line-height: 1.02;
            letter-spacing: 0;
            color: var(--ink);
        }

        .hero-copy {
            margin: 0.8rem 0 0 0;
            color: var(--muted);
            max-width: 610px;
            font-size: 1rem;
        }

        .status-card {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            min-width: max-content;
            padding: 0.65rem 0.8rem;
            border-radius: 999px;
            background: #eefbf6;
            border: 1px solid rgba(15, 118, 110, 0.18);
            color: #0d5f58;
            font-weight: 700;
            font-size: 0.88rem;
        }

        .status-card span {
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 50%;
            background: #13b981;
            box-shadow: 0 0 0 5px rgba(19, 185, 129, 0.16);
        }

        .agent-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.9rem;
            margin: 0 0 0.95rem 0;
            padding: 0.85rem 1rem;
            border-radius: 16px;
            background: rgba(22, 32, 31, 0.88);
            color: #ffffff;
            box-shadow: 0 12px 35px rgba(22, 32, 31, 0.12);
        }

        .agent-card strong {
            color: #ffffff;
            font-size: 0.95rem;
        }

        .agent-card span {
            color: #d8f3ea;
            text-align: right;
            font-size: 0.9rem;
            font-weight: 700;
        }

        .quick-prompts {
            margin-bottom: 0.8rem;
        }

        .stButton > button {
            min-height: 3.2rem;
            border-radius: 14px;
            border: 1px solid rgba(22, 32, 31, 0.11);
            background: rgba(255, 255, 255, 0.78);
            color: var(--ink);
            font-weight: 700;
            box-shadow: 0 8px 24px rgba(22, 32, 31, 0.07);
            white-space: normal;
        }

        .stButton > button:hover {
            border-color: rgba(15, 118, 110, 0.35);
            background: #ffffff;
            color: #0f766e;
        }

        [data-testid="stChatMessage"] {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 10px 28px rgba(22, 32, 31, 0.08);
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #e8f8f2;
            border-color: rgba(15, 118, 110, 0.18);
        }

        [data-testid="stChatInput"] {
            background: rgba(247, 250, 248, 0.74);
            backdrop-filter: blur(16px);
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 18px;
            border: 1px solid rgba(22, 32, 31, 0.13);
            box-shadow: 0 14px 35px rgba(22, 32, 31, 0.10);
        }

        .stDataFrame {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid var(--line);
        }

        @media (max-width: 720px) {
            .block-container {
                padding-top: 1rem;
            }

            .chat-hero {
                flex-direction: column;
                border-radius: 16px;
            }

            .chat-hero h1 {
                font-size: 2.15rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="SIH/SUS Chat", layout="wide")
    apply_style()
    render_chat()


if __name__ == "__main__":
    main()
