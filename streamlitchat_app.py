from __future__ import annotations

import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
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
    "host": "bigdata.dataiesb.com",
    "port": 5432,
    "dbname": "iesb",
    "user": "data_iesb",
    "password": "iesb",
    "connect_timeout": 20,
}

FACT_TABLE = "public.sus_aih"
DICTIONARY_PATH = Path(__file__).with_name("data_dictionary.csv")
BEDROCK_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
BEDROCK_REGION = "us-east-2"
STATEMENT_TIMEOUT_MS = 20000

PERIOD_SQL = "make_date(trim(ano_aih)::int, trim(mes_aih)::int, 1)"
PERIOD_LABEL_SQL = "lpad(trim(mes_aih), 2, '0') || '/' || trim(ano_aih)"

SUBGROUP_LABELS = {
    "0101": "Ações de promoção e prevenção em saúde",
    "0201": "Coleta de material",
    "0202": "Diagnóstico em laboratório clínico",
    "0203": "Diagnóstico por anatomia patológica e citopatologia",
    "0204": "Diagnóstico por radiologia",
    "0205": "Diagnóstico por ultrassonografia",
    "0206": "Diagnóstico por tomografia",
    "0207": "Diagnóstico por ressonância magnética",
    "0208": "Diagnóstico por medicina nuclear in vivo",
    "0209": "Diagnóstico por endoscopia",
    "0210": "Diagnóstico por radiologia intervencionista",
    "0211": "Métodos diagnósticos em especialidades",
    "0212": "Diagnóstico e procedimentos especiais em hemoterapia",
    "0214": "Diagnóstico por teste rápido",
    "0301": "Consultas, atendimentos e acompanhamentos",
    "0302": "Fisioterapia",
    "0303": "Tratamentos clínicos",
    "0304": "Tratamento em oncologia",
    "0305": "Tratamento em nefrologia",
    "0306": "Hemoterapia",
    "0307": "Tratamentos odontológicos",
    "0308": "Tratamento de lesões, envenenamentos e outros",
    "0309": "Terapias especializadas",
    "0310": "Parto e nascimento",
    "0401": "Pequenas cirurgias e cirurgias de pele",
    "0402": "Cirurgia de glândulas endócrinas",
    "0403": "Cirurgia do sistema nervoso central e periférico",
    "0404": "Cirurgia das vias aéreas superiores, face, cabeça e pescoço",
    "0405": "Cirurgia do aparelho da visão",
    "0406": "Cirurgia do aparelho circulatório",
    "0407": "Cirurgia do aparelho digestivo, órgãos anexos e parede abdominal",
    "0408": "Cirurgia do sistema osteomuscular",
    "0409": "Cirurgia do aparelho geniturinário",
    "0410": "Cirurgia de mama",
    "0411": "Cirurgia obstétrica",
    "0412": "Cirurgia torácica",
    "0413": "Cirurgia reparadora",
    "0414": "Bucomaxilofacial",
    "0415": "Outras cirurgias",
    "0416": "Cirurgia em oncologia",
    "0417": "Anestesiologia",
    "0418": "Cirurgia em nefrologia",
    "0501": "Coleta e exames para doação de órgãos",
    "0502": "Avaliação de morte encefálica",
    "0503": "Ações relacionadas à doação de órgãos",
    "0504": "Processamento de tecidos para transplante",
    "0505": "Transplante de órgãos, tecidos e células",
    "0506": "Acompanhamento e intercorrências no pré e pós-transplante",
    "0603": "Medicamentos de âmbito hospitalar",
    "0702": "Órtese, prótese e materiais especiais relacionados ao ato cirúrgico",
    "0801": "Ações relacionadas ao estabelecimento",
    "0802": "Ações relacionadas ao atendimento",
}

QTD_SUBGROUP_CODES = [
    "0101",
    "0201",
    "0202",
    "0203",
    "0204",
    "0205",
    "0206",
    "0207",
    "0208",
    "0209",
    "0210",
    "0211",
    "0212",
    "0214",
    "0301",
    "0302",
    "0303",
    "0304",
    "0305",
    "0306",
    "0307",
    "0308",
    "0309",
    "0310",
    "0401",
    "0402",
    "0403",
    "0404",
    "0405",
    "0406",
    "0407",
    "0408",
    "0409",
    "0410",
    "0411",
    "0412",
    "0413",
    "0414",
    "0415",
    "0416",
    "0418",
    "0501",
    "0502",
    "0503",
    "0504",
    "0505",
    "0506",
    "0603",
    "0702",
    "0801",
    "0802",
]
VALUE_SUBGROUP_CODES = [code for code in QTD_SUBGROUP_CODES if code != "0101"] + ["0417"]

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


@st.cache_data
def load_data_dictionary() -> pd.DataFrame:
    frame = pd.read_csv(DICTIONARY_PATH, encoding="utf-8-sig")
    frame["variavel"] = frame["variavel"].astype(str).str.lower()
    frame["tamanho_dicionario"] = frame["tamanho_dicionario"].astype("Int64")
    return frame


def dictionary_prompt_text() -> str:
    frame = load_data_dictionary()
    return "\n".join(
        f"- {row.variavel}: {row.descricao}; categoria={row.categoria}; tipo_postgresql={row.tipo_postgresql}"
        for row in frame.itertuples(index=False)
    )


def dictionary_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame[
        [
            "variavel",
            "descricao",
            "categoria",
            "tipo_postgresql",
            "tipo_dicionario",
            "tamanho_dicionario",
        ]
    ].copy()
    display["tipo_dicionario"] = display["tipo_dicionario"].astype("string").fillna("")
    display["tamanho_dicionario"] = display["tamanho_dicionario"].astype("string").fillna("")
    return display.rename(
        columns={
            "variavel": "Variável",
            "descricao": "Descrição",
            "categoria": "Categoria",
            "tipo_postgresql": "Tipo no PostgreSQL",
            "tipo_dicionario": "Tipo original",
            "tamanho_dicionario": "Tamanho original",
        }
    )


def format_number(value: Any, content: str) -> str:
    if value is None:
        return "0"
    number = float(value)
    if content == "valor_aprovado":
        return f"R$ {number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{number:,.0f}".replace(",", ".")


def metric_column(content: str) -> str:
    return "vl_total" if content == "valor_aprovado" else "qtd_total"


def metric_expr(content: str) -> str:
    return f"coalesce({metric_column(content)}, 0)"


def subgroup_codes(content: str) -> list[str]:
    return VALUE_SUBGROUP_CODES if content == "valor_aprovado" else QTD_SUBGROUP_CODES


def subgroup_column(content: str, code: str) -> str:
    return f"vl_{code}" if content == "valor_aprovado" else f"qtd_{code}"


def has_subgroup_column(content: str, code: str) -> bool:
    return code in subgroup_codes(content)


def subgroup_prompt_dictionary() -> str:
    lines: list[str] = []
    for code in QTD_SUBGROUP_CODES:
        qtd_column = f"qtd_{code}"
        value_column = f"vl_{code}" if code in VALUE_SUBGROUP_CODES else "sem coluna de valor"
        label = SUBGROUP_LABELS.get(code, code)
        lines.append(f"- {code}: quantidade={qtd_column}; valor={value_column}; nome={label}")
    return "\n".join(lines)


def validate_safe_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    compact = re.sub(r"\s+", " ", sql).strip()
    lowered = compact.lower()

    if not lowered.startswith(("select ", "with ")):
        raise ValueError("Guardrail: somente consultas SELECT ou WITH sao permitidas.")

    forbidden_words = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "copy",
        "grant",
        "revoke",
        "execute",
        "call",
        "set",
        "merge",
        "vacuum",
    ]
    if any(re.search(rf"\b{word}\b", lowered) for word in forbidden_words):
        raise ValueError("Guardrail: a consulta contem comando nao permitido.")

    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("Guardrail: comentarios ou multiplas instrucoes nao sao permitidos.")

    if FACT_TABLE.lower() not in lowered:
        raise ValueError(f"Guardrail: use somente a tabela {FACT_TABLE}.")

    return sql + ";"


def run_query(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    sql = validate_safe_sql(sql)
    conn = psycopg2.connect(**get_db_config())
    try:
        conn.set_session(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute("set local statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


@st.cache_data(ttl=600)
def cached_query(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return run_query(sql, params)


def get_bedrock_api_key() -> str | None:
    for key_name in ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_API_KEY"):
        value = os.environ.get(key_name)
        if value:
            return value.strip()

    try:
        return st.secrets.get("AWS_BEARER_TOKEN_BEDROCK") or st.secrets.get("BEDROCK_API_KEY")
    except Exception:
        return None


def get_bedrock_model_id() -> str:
    return get_secret("BEDROCK_MODEL_ID", BEDROCK_MODEL_ID) or BEDROCK_MODEL_ID


def get_bedrock_region() -> str:
    return get_secret("BEDROCK_REGION", BEDROCK_REGION) or BEDROCK_REGION


@st.cache_data(ttl=1800)
def get_periods() -> pd.DataFrame:
    return cached_query(
        f"""
        select distinct
            {PERIOD_SQL} as periodo,
            {PERIOD_LABEL_SQL} as periodo_rotulo
        from {FACT_TABLE}
        order by periodo
        """
    )


@st.cache_data(ttl=1800)
def get_ufs() -> list[str]:
    frame = cached_query(
        f"""
        select distinct trim(uf_sigla) as municipio_uf
        from {FACT_TABLE}
        where nullif(trim(uf_sigla), '') is not null
        order by municipio_uf
        """
    )
    return frame["municipio_uf"].tolist()


@st.cache_data(ttl=1800)
def get_subgroups() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subgrupo_codigo": code,
                "subgrupo_nome": SUBGROUP_LABELS.get(code, code),
            }
            for code in QTD_SUBGROUP_CODES
        ]
    )


def build_where(filters: Filters) -> tuple[str, list[Any]]:
    clauses = ["1 = 1"]
    params: list[Any] = []

    if filters.period is not None:
        clauses.append(f"{PERIOD_SQL} = %s")
        params.append(filters.period)
    elif filters.year is not None:
        clauses.append("trim(ano_aih)::int = %s")
        params.append(filters.year)

    if filters.uf:
        clauses.append("upper(trim(uf_sigla)) = %s")
        params.append(filters.uf)

    if filters.municipality:
        clauses.append("upper(trim(nome_municipio)) like upper(%s)")
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
    clauses = ["1 = 1"]

    if filters.period is not None:
        clauses.append(
            f"make_date(trim({prefix}ano_aih)::int, trim({prefix}mes_aih)::int, 1) = {sql_literal(filters.period)}"
        )
    elif filters.year is not None:
        clauses.append(f"trim({prefix}ano_aih)::int = {filters.year}")

    if filters.uf:
        clauses.append(f"upper(trim({prefix}uf_sigla)) = {sql_literal(filters.uf)}")

    if filters.municipality:
        clauses.append(f"upper(trim({prefix}nome_municipio)) like upper({sql_literal('%' + filters.municipality + '%')})")

    return "\n  and ".join(clauses)


def sql_for_base_info() -> str:
    return f"""select
    count(*) as linhas,
    count(distinct {PERIOD_SQL}) as periodos,
    min({PERIOD_SQL}) as primeiro_periodo,
    max({PERIOD_SQL}) as ultimo_periodo,
    count(distinct nullif(trim(codigo_municipio), '')) as municipios
from {FACT_TABLE};"""


def sql_for_metric_summary(filters: Filters) -> str:
    total_column = metric_expr(filters.content)
    return f"""select
    count(*) as linhas,
    count(distinct nullif(trim(codigo_municipio), '')) as municipios,
    sum({total_column}) as total,
    avg({total_column}) as media,
    max({total_column}) as maior
from {FACT_TABLE}
where {build_display_where(filters)};"""


def sql_for_timeline(filters: Filters) -> str:
    query_filters = Filters(content=filters.content, uf=filters.uf, municipality=filters.municipality)
    total_column = metric_expr(filters.content)
    return f"""select
    {PERIOD_SQL} as periodo,
    {PERIOD_LABEL_SQL} as periodo_rotulo,
    sum({total_column}) as total
from {FACT_TABLE}
where {build_display_where(query_filters)}
group by periodo, periodo_rotulo
order by periodo;"""


def sql_for_top_municipalities(filters: Filters, limit: int) -> str:
    total_column = metric_expr(filters.content)
    return f"""select
    trim(nome_municipio) as municipio_nome,
    trim(uf_sigla) as municipio_uf,
    sum({total_column}) as total
from {FACT_TABLE}
where {build_display_where(filters)}
group by municipio_nome, municipio_uf
order by total desc
limit {limit};"""


def sql_for_top_ufs(filters: Filters, limit: int) -> str:
    total_column = metric_expr(filters.content)
    return f"""select
    trim(uf_sigla) as municipio_uf,
    sum({total_column}) as total
from {FACT_TABLE}
where {build_display_where(filters)}
  and nullif(trim(uf_sigla), '') is not null
group by municipio_uf
order by total desc
limit {limit};"""


def sql_for_top_subgroups(filters: Filters, limit: int) -> str:
    values_sql = ",\n        ".join(
        f"('{code}', '{SUBGROUP_LABELS.get(code, code)}', coalesce(t.{subgroup_column(filters.content, code)}, 0))"
        for code in subgroup_codes(filters.content)
    )
    return f"""select
    subgrupo_codigo,
    subgrupo_nome,
    sum(valor) as total
from {FACT_TABLE} t
cross join lateral (
    values
        {values_sql}
) as s(subgrupo_codigo, subgrupo_nome, valor)
where {build_display_where(filters, alias="t")}
group by subgrupo_codigo, subgrupo_nome
order by total desc
limit {limit};"""


def sql_for_subgroup_total(filters: Filters, code: str) -> str:
    total_column = subgroup_column(filters.content, code)
    subgrupo_nome = SUBGROUP_LABELS.get(code, code)
    return f"""select
    {sql_literal(code)} as subgrupo_codigo,
    {sql_literal(subgrupo_nome)} as subgrupo_nome,
    sum(coalesce({total_column}, 0)) as total
from {FACT_TABLE}
where {build_display_where(filters)};"""


def sql_for_subgroup_totals(filters: Filters, codes: list[str]) -> str:
    values_sql = ",\n        ".join(
        f"('{code}', '{SUBGROUP_LABELS.get(code, code)}', coalesce(t.{subgroup_column(filters.content, code)}, 0))"
        for code in codes
    )
    return f"""with resultados as (
    select
        s.subgrupo_codigo,
        s.subgrupo_nome,
        sum(s.valor) as total
    from {FACT_TABLE} t
    cross join lateral (
        values
        {values_sql}
    ) as s(subgrupo_codigo, subgrupo_nome, valor)
    where {build_display_where(filters, alias="t")}
    group by s.subgrupo_codigo, s.subgrupo_nome
)
select subgrupo_codigo, subgrupo_nome, total
from (
    select subgrupo_codigo, subgrupo_nome, total from resultados
    union all
    select 'TOTAL', 'Total combinado', sum(total) from resultados
) as resposta
order by case when subgrupo_codigo = 'TOTAL' then 1 else 0 end, subgrupo_codigo;"""


def get_schema_prompt() -> str:
    return f"""
Você é um agente gerador de SQL para PostgreSQL.
Gere apenas consultas SELECT seguras sobre a base SIH/SUS.

Tabelas disponiveis:
1. {FACT_TABLE}
   - ano_aih CHAR: ano da AIH
   - mes_aih CHAR: mes da AIH
   - codigo_municipio TEXT
   - nome_municipio TEXT
   - uf_sigla CHAR(2)
   - qtd_total INTEGER: quantidade aprovada
   - vl_total NUMERIC: valor aprovado
   - colunas qtd_XXXX e vl_XXXX: totais por subgrupo de procedimento

Dicionário de subgrupos:
{subgroup_prompt_dictionary()}

Dicionário completo das colunas da tabela:
{dictionary_prompt_text()}

Regras obrigatorias:
- Responda somente em JSON valido.
- Use somente SELECT ou WITH.
- Nunca use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, COPY ou comandos administrativos.
- Use somente a tabela {FACT_TABLE}.
- Para "quantidade aprovada", use qtd_total.
- Para "valor aprovado", use vl_total.
- Use coalesce(qtd_total, 0) ou coalesce(vl_total, 0) ao somar ou ordenar totais.
- Para subgrupos, use a coluna do dicionário. Exemplo: valor aprovado do subgrupo 0204 usa vl_0204; quantidade aprovada do subgrupo 0204 usa qtd_0204.
- Para ranking de subgrupos, faça um unpivot com cross join lateral values usando as colunas qtd_XXXX ou vl_XXXX.
- Use somente nomes de colunas presentes no dicionário completo.
- As colunas qtd_ representam quantidades e as colunas vl_ representam valores monetários.
- Para datas mensais, use make_date(trim(ano_aih)::int, trim(mes_aih)::int, 1).
- Janeiro de 2026 deve ser filtrado como trim(ano_aih)::int = 2026 e trim(mes_aih)::int = 1.
- Para municipio, use trim(nome_municipio). Para UF, use trim(uf_sigla).
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

    if " limit " not in lowered and not any(term in lowered for term in ["sum(", "count(", "avg(", "max(", "min("]):
        sql = f"{sql}\nlimit 30"

    return validate_safe_sql(sql)


def call_bedrock_for_sql(question: str) -> dict[str, Any]:
    api_key = get_bedrock_api_key()
    if not api_key:
        raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK nao configurada.")

    model_id = get_bedrock_model_id()
    region = get_bedrock_region()
    endpoint = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model_id}/converse"
    prompt = f"{get_schema_prompt()}\n\nPergunta do usuario:\n{question}"
    payload = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {
            "temperature": 0,
            "maxTokens": 1200,
        },
    }
    response = requests.post(
        endpoint,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    text = data["output"]["message"]["content"][0]["text"]
    parsed = extract_json_from_text(text)
    parsed["sql"] = validate_generated_sql(parsed["sql"])
    parsed["chart_type"] = parsed.get("chart_type") or "none"
    parsed["content_key"] = parsed.get("content_key") or detect_content(question)
    return parsed


def summarize_llm_result(question: str, frame: pd.DataFrame, parsed: dict[str, Any]) -> str:
    content_key = parsed.get("content_key", detect_content(question))
    if frame.empty:
        return "O Amazon Bedrock gerou a consulta, mas não encontrei registros para esses filtros."

    first = frame.iloc[0]
    if "municipio_nome" in frame.columns and "total" in frame.columns:
        uf = f" ({first['municipio_uf']})" if "municipio_uf" in frame.columns else ""
        return f"Segundo a consulta gerada pelo Amazon Bedrock, o principal município foi {first['municipio_nome']}{uf}, com {format_number(first['total'], content_key)}."

    if "municipio_uf" in frame.columns and "total" in frame.columns:
        return f"Segundo a consulta gerada pelo Amazon Bedrock, a principal UF foi {first['municipio_uf']}, com {format_number(first['total'], content_key)}."

    if "subgrupo_nome" in frame.columns and "total" in frame.columns:
        return f"Segundo a consulta gerada pelo Amazon Bedrock, o principal subgrupo foi {first['subgrupo_nome']}, com {format_number(first['total'], content_key)}."

    if "total" in frame.columns and len(frame) == 1:
        return f"Segundo a consulta gerada pelo Amazon Bedrock, o total encontrado foi {format_number(first['total'], content_key)}."

    explanation = parsed.get("explanation") or "Consulta gerada pelo Amazon Bedrock."
    return f"{explanation} Retornei {len(frame)} linha(s) para análise."


def metric_summary(filters: Filters) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    total_column = metric_expr(filters.content)
    return cached_query(
        f"""
        select
            count(*) as linhas,
            count(distinct nullif(trim(codigo_municipio), '')) as municipios,
            sum({total_column}) as total,
            avg({total_column}) as media,
            max({total_column}) as maior
        from {FACT_TABLE}
        where {where_sql}
        """,
        tuple(params),
    )


def timeline(filters: Filters) -> pd.DataFrame:
    query_filters = Filters(content=filters.content, uf=filters.uf, municipality=filters.municipality)
    where_sql, params = build_where(query_filters)
    total_column = metric_expr(filters.content)
    return cached_query(
        f"""
        select
            {PERIOD_SQL} as periodo,
            {PERIOD_LABEL_SQL} as periodo_rotulo,
            sum({total_column}) as total
        from {FACT_TABLE}
        where {where_sql}
        group by periodo, periodo_rotulo
        order by periodo
        """,
        tuple(params),
    )


def top_municipalities(filters: Filters, limit: int = 15) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    total_column = metric_expr(filters.content)
    return cached_query(
        f"""
        select
            trim(nome_municipio) as municipio_nome,
            trim(uf_sigla) as municipio_uf,
            sum({total_column}) as total
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
    total_column = metric_expr(filters.content)
    return cached_query(
        f"""
        select trim(uf_sigla) as municipio_uf, sum({total_column}) as total
        from {FACT_TABLE}
        where {where_sql}
          and nullif(trim(uf_sigla), '') is not null
        group by municipio_uf
        order by total desc
        """,
        tuple(params),
    )


def top_subgroups(filters: Filters, limit: int = 15) -> pd.DataFrame:
    values_sql = ",\n".join(
        f"('{code}', '{SUBGROUP_LABELS.get(code, code)}', coalesce(t.{subgroup_column(filters.content, code)}, 0))"
        for code in subgroup_codes(filters.content)
    )
    where_sql, params = build_where(filters)
    return cached_query(
        f"""
        select
            d.subgrupo_codigo,
            d.subgrupo_nome,
            sum(d.valor) as total
        from {FACT_TABLE} t
        cross join lateral (
            values
            {values_sql}
        ) as d(subgrupo_codigo, subgrupo_nome, valor)
        where {where_sql}
        group by d.subgrupo_codigo, d.subgrupo_nome
        order by total desc
        limit %s
        """,
        tuple(params + [limit]),
    )


def subgroup_total(filters: Filters, code: str) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    total_column = subgroup_column(filters.content, code)
    subgrupo_nome = SUBGROUP_LABELS.get(code, code)
    return cached_query(
        f"""
        select
            %s as subgrupo_codigo,
            %s as subgrupo_nome,
            sum(coalesce({total_column}, 0)) as total
        from {FACT_TABLE}
        where {where_sql}
        """,
        tuple([code, subgrupo_nome] + params),
    )


def subgroup_totals(filters: Filters, codes: list[str]) -> pd.DataFrame:
    values_sql = ",\n".join(
        f"('{code}', '{SUBGROUP_LABELS.get(code, code)}', coalesce(t.{subgroup_column(filters.content, code)}, 0))"
        for code in codes
    )
    where_sql, params = build_where(filters)
    return cached_query(
        f"""
        with resultados as (
            select
                s.subgrupo_codigo,
                s.subgrupo_nome,
                sum(s.valor) as total
            from {FACT_TABLE} t
            cross join lateral (
                values
                {values_sql}
            ) as s(subgrupo_codigo, subgrupo_nome, valor)
            where {where_sql}
            group by s.subgrupo_codigo, s.subgrupo_nome
        )
        select subgrupo_codigo, subgrupo_nome, total
        from (
            select subgrupo_codigo, subgrupo_nome, total from resultados
            union all
            select 'TOTAL', 'Total combinado', sum(total) from resultados
        ) as resposta
        order by case when subgrupo_codigo = 'TOTAL' then 1 else 0 end, subgrupo_codigo
        """,
        tuple(params),
    )


def data_sample(filters: Filters, limit: int = 200) -> pd.DataFrame:
    where_sql, params = build_where(filters)
    total_column = metric_expr(filters.content)
    return cached_query(
        f"""
        select
            {PERIOD_LABEL_SQL} as periodo_rotulo,
            {sql_literal(filters.content)} as conteudo,
            trim(codigo_municipio) as municipio_codigo,
            trim(nome_municipio) as municipio_nome,
            trim(uf_sigla) as municipio_uf,
            {total_column} as total_linha
        from {FACT_TABLE}
        where {where_sql}
        order by {PERIOD_SQL}, municipio_uf, municipio_nome
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


def detect_subgroup_code(question: str) -> str | None:
    for code in re.findall(r"\b\d{4}\b", question):
        if code in SUBGROUP_LABELS:
            return code
    return None


@st.cache_data
def subgroup_name_aliases() -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = {code: {normalize_text(label)} for code, label in SUBGROUP_LABELS.items()}
    frame = load_data_dictionary()
    subgroup_rows = frame[frame["variavel"].str.match(r"^(qtd|vl)_\d{4}$")]
    prefix_pattern = re.compile(
        r"^(?:quantidade|valor)\s+(?:de|da|das|do|dos)?\s*",
    )

    for row in subgroup_rows.itertuples(index=False):
        code = row.variavel.split("_", 1)[1]
        description = normalize_text(row.descricao)
        aliases.setdefault(code, set()).add(description)
        aliases[code].add(prefix_pattern.sub("", description).strip())

    return {
        code: sorted((alias for alias in values if len(alias) >= 4), key=len, reverse=True)
        for code, values in aliases.items()
    }


def detect_subgroup_names(question: str) -> list[str]:
    text = normalize_text(question)
    matches: list[tuple[int, str]] = []
    for code, aliases in subgroup_name_aliases().items():
        for alias in aliases:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text):
                matches.append((len(alias), code))
                break

    if matches:
        longest = max(length for length, _ in matches)
        return sorted({code for length, code in matches if length == longest})

    ignored = {
        "a",
        "as",
        "aprovada",
        "aprovadas",
        "aprovado",
        "aprovados",
        "de",
        "do",
        "dos",
        "da",
        "das",
        "em",
        "no",
        "nos",
        "na",
        "nas",
        "o",
        "os",
        "qual",
        "quais",
        "quantidade",
        "total",
        "valor",
    }

    def token_key(value: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", value)) - ignored
        return {
            token[:-1] if token.endswith("s") and len(token) > 5 else token
            for token in tokens
        }

    question_tokens = token_key(text)
    if not question_tokens:
        return []

    scores: dict[str, tuple[int, float, int]] = {}
    for code, aliases in subgroup_name_aliases().items():
        best_score = (0, 0.0, 0)
        for alias in aliases:
            alias_tokens = token_key(alias)
            if not alias_tokens:
                continue
            overlap = len(alias_tokens & question_tokens)
            score = (overlap, overlap / len(alias_tokens), len(alias))
            if score > best_score:
                best_score = score
        if best_score[0] > 0:
            scores[code] = best_score

    if not scores:
        return []

    if len(question_tokens) == 1:
        return sorted(scores)

    max_overlap = max(score[0] for score in scores.values())
    candidates = {
        code: score for code, score in scores.items() if score[0] == max_overlap
    }
    max_ratio = max(score[1] for score in candidates.values())
    return sorted(
        code for code, score in candidates.items() if score[1] == max_ratio
    )


def detect_subgroup_name(question: str) -> str | None:
    matches = detect_subgroup_names(question)
    return matches[0] if matches else None


def answer_dictionary_question(
    question: str,
) -> tuple[str, pd.DataFrame | None, str | None, str | None, str] | None:
    text = normalize_text(question)
    intent_terms = [
        "dicionario",
        "o que significa",
        "o que e",
        "qual campo",
        "quais campos",
        "qual coluna",
        "quais colunas",
        "qual variavel",
        "quais variaveis",
        "quantas colunas",
        "quantos campos",
        "quantas variaveis",
        "descricao",
        "tipo de dado",
        "tipo no banco",
        "tamanho",
        "representa",
        "serve para",
        "diferenca",
    ]
    if not any(term in text for term in intent_terms):
        return None

    frame = load_data_dictionary().copy()
    variables = frame["variavel"].tolist()
    exact_variables = [
        variable
        for variable in sorted(variables, key=len, reverse=True)
        if re.search(rf"(?<!\w){re.escape(variable)}(?!\w)", text)
    ]

    if exact_variables:
        matches = frame[frame["variavel"].isin(exact_variables)]
    else:
        code_match = re.search(r"\b(\d{2}|\d{4})\b", text)
        matches = frame.iloc[0:0]
        if code_match:
            code = code_match.group(1)
            matches = frame[frame["variavel"].str.endswith(f"_{code}")]
            if "valor" in text:
                matches = matches[matches["variavel"].str.startswith("vl_")]
            elif "quantidade" in text or "qtd" in text:
                matches = matches[matches["variavel"].str.startswith("qtd_")]

    category_terms = {
        "Valor": ["valor", "monetario", "financeiro"],
        "Quantidade": ["quantidade", "qtd"],
        "Localização": ["localizacao", "municipio", "regiao", "estado", "uf"],
        "População": ["populacao", "habitantes", "fpm"],
        "Coordenadas": ["coordenadas", "latitude", "longitude"],
        "Período": ["periodo", "ano", "mes"],
    }
    selected_category = next(
        (
            category
            for category, terms in category_terms.items()
            if any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)
        ),
        None,
    )
    is_count_question = any(
        term in text for term in ["quantas colunas", "quantos campos", "quantas variaveis"]
    )

    if is_count_question and matches.empty:
        matches = frame
        if selected_category:
            matches = matches[matches["categoria"] == selected_category]

    if matches.empty:
        candidates = frame
        if selected_category:
            candidates = candidates[candidates["categoria"] == selected_category]

        stop_words = {
            "dicionario",
            "significa",
            "descricao",
            "campo",
            "campos",
            "coluna",
            "colunas",
            "variavel",
            "variaveis",
            "representa",
            "representam",
            "serve",
            "tipo",
            "dado",
            "dados",
            "tamanho",
            "qual",
            "quais",
            "quantas",
            "quantos",
            "todos",
            "todas",
            "aprovado",
            "aprovada",
            "valor",
            "quantidade",
            "monetario",
            "financeiro",
            "localizacao",
            "populacao",
            "coordenadas",
            "periodo",
            "banco",
            "tabela",
            "para",
            "pela",
            "pelo",
            "com",
            "uma",
            "dos",
            "das",
            "que",
            "sao",
        }
        tokens = {
            token
            for token in re.findall(r"[a-z0-9_]+", text)
            if len(token) >= 3 and token not in stop_words
        }
        for terms in category_terms.values():
            tokens.difference_update(terms)

        if tokens:
            searchable = candidates.apply(
                lambda row: normalize_text(f"{row['variavel']} {row['descricao']} {row['categoria']}"),
                axis=1,
            )
            scores = searchable.map(lambda value: sum(token in value for token in tokens))
            if scores.max() > 0:
                matches = candidates[scores == scores.max()]
        elif selected_category:
            matches = candidates

    if matches.empty:
        return (
            "Não encontrei uma variável correspondente no dicionário da public.sus_aih. Tente informar o nome da coluna, como vl_0204, ou uma descrição, como radiologia.",
            None,
            None,
            None,
            "Dicionário de dados",
        )

    matches = matches.sort_values("ordem")
    if is_count_question:
        category_suffix = f" na categoria {selected_category}" if selected_category else ""
        return (
            f"O dicionário possui {len(matches)} variáveis{category_suffix}.",
            dictionary_display_frame(matches.head(30)),
            None,
            None,
            "Dicionário de dados",
        )

    if "diferenca" in text and len(matches) > 1:
        details = " ".join(
            f"`{row.variavel}` significa **{row.descricao}** e usa `{row.tipo_postgresql}`."
            for row in matches.itertuples(index=False)
        )
        answer = f"A diferença é a seguinte: {details}"
    elif len(matches) == 1:
        row = matches.iloc[0]
        size = (
            f", tamanho original {int(row['tamanho_dicionario'])}"
            if pd.notna(row["tamanho_dicionario"])
            else ""
        )
        answer = (
            f"A variável `{row['variavel']}` significa **{row['descricao']}**. "
            f"Ela pertence à categoria {row['categoria']} e usa o tipo "
            f"`{row['tipo_postgresql']}` no PostgreSQL{size}."
        )
    else:
        answer = f"Encontrei {len(matches)} variáveis relacionadas no dicionário da `public.sus_aih`."

    return (
        answer,
        dictionary_display_frame(matches.head(30)),
        None,
        None,
        "Dicionário de dados",
    )


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
        parts.append("todos os períodos")
    if filters.uf:
        parts.append(filters.uf)
    if filters.municipality:
        parts.append(f"município contendo '{filters.municipality}'")
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
    explicit_subgroup_code = detect_subgroup_code(question)
    matched_subgroup_codes = (
        [explicit_subgroup_code]
        if explicit_subgroup_code
        else detect_subgroup_names(question)
    )

    if any(term in text for term in ["periodos", "base", "registros", "linhas"]):
        frame = cached_query(
            f"""
            select
                count(*) as linhas,
                count(distinct {PERIOD_SQL}) as periodos,
                min({PERIOD_SQL}) as primeiro_periodo,
                max({PERIOD_SQL}) as ultimo_periodo,
                count(distinct nullif(trim(codigo_municipio), '')) as municipios
            from {FACT_TABLE}
            """
        )
        row = frame.iloc[0]
        return (
            f"A base tem {int(row.linhas):,} linhas, {int(row.periodos)} períodos e "
            f"{int(row.municipios):,} municípios/linhas municipais.".replace(",", "."),
            frame,
            None,
            sql_for_base_info(),
            "Fallback por regras",
        )

    if matched_subgroup_codes:
        available_codes = [
            code for code in matched_subgroup_codes if has_subgroup_column(filters.content, code)
        ]
        if not available_codes:
            label = CONTENT_LABELS[filters.content]
            names = ", ".join(
                f"{code} - {SUBGROUP_LABELS.get(code, code)}"
                for code in matched_subgroup_codes
            )
            return (
                f"Não existe coluna de {label.lower()} para {names} nesta tabela.",
                None,
                None,
                None,
                "Fallback por regras",
            )

        if len(available_codes) == 1:
            subgroup_code = available_codes[0]
            frame = subgroup_total(filters, subgroup_code)
            total = frame.iloc[0]["total"] if not frame.empty else 0
            subgrupo_nome = SUBGROUP_LABELS.get(subgroup_code, subgroup_code)
            message = (
                f"O total do subgrupo {subgroup_code} - {subgrupo_nome} para "
                f"{describe_filters(filters)} é {format_number(total, filters.content)}."
            )
            return message, frame, None, sql_for_subgroup_total(filters, subgroup_code), "Fallback por regras"

        frame = subgroup_totals(filters, available_codes)
        total_rows = frame[frame["subgrupo_codigo"] == "TOTAL"]
        combined_total = total_rows.iloc[0]["total"] if not total_rows.empty else 0
        names = ", ".join(
            f"{code} - {SUBGROUP_LABELS.get(code, code)}" for code in available_codes
        )
        message = (
            f"Encontrei {len(available_codes)} subgrupos relacionados: {names}. "
            f"O total combinado para {describe_filters(filters)} é "
            f"{format_number(combined_total, filters.content)}."
        )
        return (
            message,
            frame,
            None,
            sql_for_subgroup_totals(filters, available_codes),
            "Fallback por regras",
        )

    if "subgrupo" in text or "procedimento" in text:
        frame = top_subgroups(filters, limit=limit)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = (
            f"O maior subgrupo para {describe_filters(filters)} foi "
            f"{frame.iloc[0]['subgrupo_nome']} com {format_number(total, filters.content)}."
            if not frame.empty
            else "Não encontrei resultado para esses filtros."
        )
        return message, frame, "bar", sql_for_top_subgroups(filters, limit), "Fallback por regras"

    if any(term in text for term in ["estado", "uf", "ufs"]):
        frame = top_ufs(filters)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = (
            f"A UF com maior total para {describe_filters(filters)} foi "
            f"{frame.iloc[0]['municipio_uf']} com {format_number(total, filters.content)}."
            if not frame.empty
            else "Não encontrei resultado para esses filtros."
        )
        return message, frame.head(limit), "bar", sql_for_top_ufs(filters, limit), "Fallback por regras"

    if any(term in text for term in ["evolucao", "mensal", "por mes", "serie"]):
        frame = timeline(filters)
        return f"Evolução mensal para {describe_filters(filters)}.", frame, "line", sql_for_timeline(filters), "Fallback por regras"

    if any(term in text for term in ["maior", "ranking", "top", "municipios", "cidades"]):
        frame = top_municipalities(filters, limit=limit)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = (
            f"O maior município para {describe_filters(filters)} foi "
            f"{frame.iloc[0]['municipio_nome']} ({frame.iloc[0]['municipio_uf']}) "
            f"com {format_number(total, filters.content)}."
            if not frame.empty
            else "Não encontrei resultado para esses filtros."
        )
        return message, frame, "bar", sql_for_top_municipalities(filters, limit), "Fallback por regras"

    if "total" in text or "soma" in text:
        frame = metric_summary(filters)
        total = frame.iloc[0]["total"] if not frame.empty else 0
        message = f"O total para {describe_filters(filters)} é {format_number(total, filters.content)}."
        return message, frame, None, sql_for_metric_summary(filters), "Fallback por regras"

    if filters.municipality:
        frame = top_municipalities(filters, limit=limit)
        return f"Resultado para {describe_filters(filters)}.", frame, "bar", sql_for_top_municipalities(filters, limit), "Fallback por regras"

    return (
        "Posso responder totais, maiores municípios, ranking por UF, ranking por subgrupo "
        "e evolução mensal. Exemplo: total de quantidade aprovada em janeiro de 2026.",
        None,
        None,
        None,
        "Fallback por regras",
    )


def answer_question(question: str) -> tuple[str, pd.DataFrame | None, str | None, str | None, str]:
    dictionary_answer = answer_dictionary_question(question)
    if dictionary_answer is not None:
        return dictionary_answer

    if get_bedrock_api_key():
        try:
            parsed = call_bedrock_for_sql(question)
            frame = cached_query(parsed["sql"])
            message = summarize_llm_result(question, frame, parsed)
            return (
                message,
                frame,
                parsed.get("chart_type"),
                parsed["sql"],
                f"Amazon Bedrock ({get_bedrock_model_id()})",
            )
        except Exception as exc:
            fallback_answer, frame, chart_type, sql_query, _ = answer_question_rules(question)
            message = (
                f"Não consegui usar o Amazon Bedrock nesta pergunta, então respondi pelo fallback por regras. "
                f"{fallback_answer}"
            )
            return message, frame, chart_type, sql_query, f"Fallback por regras - Bedrock indisponível: {exc}"

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
    with st.expander("Consulta SQL usada", expanded=True):
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


def render_dictionary() -> None:
    frame = load_data_dictionary()
    st.markdown(
        """
        <section class="chat-hero dictionary-hero">
            <div>
                <p class="eyebrow">PUBLIC.SUS_AIH</p>
                <h1>Dicionário de Dados</h1>
                <p class="hero-copy">Definições das variáveis da produção hospitalar SIH/SUS, conciliadas com as colunas reais do PostgreSQL.</p>
            </div>
            <div class="status-card">
                <span></span>
                135 variáveis
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    total_quantity = int((frame["categoria"] == "Quantidade").sum())
    total_value = int((frame["categoria"] == "Valor").sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Variáveis", len(frame))
    col2.metric("Campos de quantidade", total_quantity)
    col3.metric("Campos de valor", total_value)

    search_col, category_col = st.columns([1.7, 0.8])
    with search_col:
        search = st.text_input(
            "Buscar no dicionário",
            placeholder="Ex.: vl_0204, radiologia, município ou população",
        )
    with category_col:
        categories = ["Todas"] + sorted(frame["categoria"].dropna().unique().tolist())
        category = st.selectbox("Categoria", categories)

    filtered = frame
    if category != "Todas":
        filtered = filtered[filtered["categoria"] == category]
    if search.strip():
        normalized_search = normalize_text(search.strip())
        searchable = filtered.apply(
            lambda row: normalize_text(
                f"{row['variavel']} {row['descricao']} {row['categoria']} {row['tipo_postgresql']}"
            ),
            axis=1,
        )
        filtered = filtered[searchable.str.contains(re.escape(normalized_search), regex=True)]

    st.caption(f"{len(filtered)} variável(is) encontrada(s)")
    st.dataframe(
        dictionary_display_frame(filtered),
        width="stretch",
        height=540,
        hide_index=True,
        column_config={
            "Variável": st.column_config.TextColumn(width="medium"),
            "Descrição": st.column_config.TextColumn(width="large"),
            "Categoria": st.column_config.TextColumn(width="small"),
            "Tipo no PostgreSQL": st.column_config.TextColumn(width="medium"),
        },
    )

    st.subheader("Convenções das variáveis")
    st.markdown(
        """
        - `qtd_XXXX`: quantidade aprovada do grupo ou subgrupo de procedimento.
        - `vl_XXXX`: valor aprovado do grupo ou subgrupo de procedimento.
        - `qtd_total` e `vl_total`: totais gerais de quantidade e valor.
        - Os códigos com dois dígitos representam grupos; os códigos com quatro dígitos representam subgrupos.
        """
    )
    st.caption("Fonte: dicionário SUS-AIH-2026-04-08 e estrutura atual da tabela public.sus_aih.")


def render_chat() -> None:
    st.title("Agente gerador de SQL")

    if st.session_state.get("chat_ui_version") != "sql-only-v1":
        st.session_state.messages = []
        st.session_state.chat_ui_version = "sql-only-v1"

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            frame = message.get("frame")
            sql_query = message.get("sql_query")
            render_sql_query(sql_query)
            if isinstance(frame, pd.DataFrame):
                st.table(frame)
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Digite sua pergunta")
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
        }
        st.session_state.messages.append(response)
        with st.chat_message("assistant"):
            st.write(answer)
            render_sql_query(sql_query)
            if isinstance(frame, pd.DataFrame):
                st.table(frame)


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
            color: var(--ink) !important;
            font-weight: 700;
            box-shadow: 0 8px 24px rgba(22, 32, 31, 0.07);
            white-space: normal;
        }

        .stButton > button *,
        .stButton > button p {
            color: var(--ink) !important;
        }

        .stButton > button:hover {
            border-color: rgba(15, 118, 110, 0.35);
            background: #ffffff;
            color: #0f766e !important;
        }

        .stButton > button:hover *,
        .stButton > button:hover p {
            color: #0f766e !important;
        }

        [data-testid="stChatMessage"] {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.94);
            color: var(--ink) !important;
            box-shadow: 0 10px 28px rgba(22, 32, 31, 0.08);
        }

        [data-testid="stChatMessage"] *,
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
            color: var(--ink) !important;
        }

        [data-testid="stChatMessage"] small,
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"],
        [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] * {
            color: #42524f !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #e8f8f2;
            border-color: rgba(15, 118, 110, 0.18);
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span {
            color: var(--ink) !important;
        }

        [data-testid="stChatInput"] {
            background: rgba(247, 250, 248, 0.74);
            backdrop-filter: blur(16px);
        }

        [data-testid="stChatInput"] textarea {
            border-radius: 18px;
            border: 1px solid rgba(22, 32, 31, 0.13);
            box-shadow: 0 14px 35px rgba(22, 32, 31, 0.10);
            color: var(--ink) !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #697874 !important;
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
    st.set_page_config(page_title="Agente gerador de SQL", layout="wide")
    apply_style()
    render_chat()


if __name__ == "__main__":
    main()
