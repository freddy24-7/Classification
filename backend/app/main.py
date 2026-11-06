"""FastAPI app: lists the topics, runs them with page settings, serves code references and PDFs."""
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .code_ref import code_reference
from .dataset import DESCRIPTIONS, column_choices, get_df
from .pdf import build_pdf
from .registry import BY_ID, GROUPS, TOPICS

app = FastAPI(title="Classification Lab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5174").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _get_topic(topic_id: str):
    topic = BY_ID.get(topic_id)
    if not topic:
        raise HTTPException(404, f"Unknown topic '{topic_id}'")
    return topic


def _parse_params(topic, query) -> dict:
    """Query string -> validated keyword arguments for the topic's function."""
    choices = column_choices(get_df())
    kwargs = {}
    for p in topic.params:
        raw = query.get(p["name"])
        if raw is None or raw == "":
            kwargs[p["name"]] = p["default"]
            continue
        kind = p["type"]
        if kind == "column":
            if raw not in choices[p["columns"]]:
                raise HTTPException(422, f"{p['label']}: '{raw}' is not one of {choices[p['columns']]}")
            kwargs[p["name"]] = raw
        elif kind == "select":
            allowed = {str(o): o for o in p["options"]}
            if raw not in allowed:
                raise HTTPException(422, f"{p['label']}: '{raw}' is not one of {list(allowed)}")
            kwargs[p["name"]] = allowed[raw]
        elif kind == "bool":
            kwargs[p["name"]] = raw.lower() in ("1", "true", "yes", "on")
    return kwargs


def _run(topic, kwargs):
    try:
        return topic.func(get_df(), **kwargs)
    except ValueError as e:
        raise HTTPException(422, str(e))


def _public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _param_specs(topic):
    choices = column_choices(get_df())
    specs = []
    for p in topic.params:
        spec = dict(p)
        if p["type"] == "column":
            spec["options"] = choices[p["columns"]]
        specs.append(spec)
    return specs


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


@app.get("/api/dataset")
def dataset_info():
    df = get_df()
    return {
        "name": "diabetes.csv",
        "rows": len(df),
        "columns": df.columns.tolist(),
        "descriptions": DESCRIPTIONS,
        "head": df.head(10).to_dict(orient="records"),
    }


@app.get("/api/topics")
def list_topics():
    return [
        {"id": key, "title": label,
         "topics": [{"id": t.id, "number": t.number, "title": t.title, "summary": t.summary}
                    for t in TOPICS if t.group == key]}
        for key, label in GROUPS.items()
    ]


@app.get("/api/topics/{topic_id}")
def get_topic(topic_id: str, request: Request):
    t = _get_topic(topic_id)
    kwargs = _parse_params(t, request.query_params)
    return {
        "id": t.id, "number": t.number, "title": t.title, "summary": t.summary, "group": t.group,
        "what": t.what, "why": t.why, "how": t.how, "fix": t.fix, "fix_title": t.fix_title, "pitfalls": t.pitfalls,
        "params": _param_specs(t), "values": kwargs,
        "result": _public(_run(t, kwargs)),
        "code": [code_reference(f) for f in [t.func] + t.helpers],
    }


@app.get("/api/topics/{topic_id}/pdf")
def get_topic_pdf(topic_id: str, request: Request):
    t = _get_topic(topic_id)
    kwargs = _parse_params(t, request.query_params)
    pdf = build_pdf(t, _run(t, kwargs), kwargs)
    disposition = "attachment" if request.query_params.get("download") == "true" else "inline"
    filename = f"{t.number.replace('.', '_')}_{t.id}.pdf"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'{disposition}; filename="{filename}"'})
