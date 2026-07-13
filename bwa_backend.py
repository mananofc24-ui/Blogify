from __future__ import annotations

import operator
import os
import re
import time 
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from datetime import datetime

def _log(node: str, message: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{node}] {message}")
    
load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema 
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str
    
    #UI 
    include_images : bool 

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    image_results: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatOllama(model="llama3.2:latest",
    temperature=0)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """
You are a routing module for a blog-writing agent.

Decide whether internet research is required.

Use these rules.

closed_book
- Programming concepts
- Algorithms
- Mathematics
- Data structures
- Tutorials
- Historical facts
- General explanations

hybrid
- Products
- Cars
- Phones
- CPUs
- GPUs
- AI models
- Frameworks
- Companies
- Libraries
- Comparisons
- Anything containing a year
- Anything containing versions
- Anything that may have changed recently

open_book
- Latest
- News
- This week
- Today
- Yesterday
- New release
- Breaking
- Market updates
- Price changes
- Policies

If the topic contains:
- a year
- "best"
- "top"
- "latest"
- "new"
- "vs"
- rankings
- products

THEN choose at least hybrid.

If needs_research=True,
generate 3–6 search queries.

Return only RouterDecision.
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)

    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=f"""
Topic: {state['topic']}

As-of date: {state['as_of']}
"""
            ),
        ]
    )

    # ----------------------------------------------------
    # Simple safeguard for topics that obviously need research
    # ----------------------------------------------------
    topic = state["topic"].lower()

    keywords = [
        "best",
        "top",
        "latest",
        "new",
        "2025",
        "2026",
        "2027",
        "vs",
        "comparison",
        "review",
        "price",
        "cars",
        "phones",
        "laptops",
    ]

    if (not decision.needs_research) and any(k in topic for k in keywords):

        decision.needs_research = True
        decision.mode = "hybrid"

        if not decision.queries:
            decision.queries = [
                state["topic"],
                f"Latest {state['topic']}",
                f"{state['topic']} review",
            ]

    # ----------------------------------------------------
    # Set recency window
    # ----------------------------------------------------
    if decision.mode == "open_book":
        recency_days = 7

    elif decision.mode == "hybrid":
        recency_days = 45

    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
from langchain_tavily import TavilySearch


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        print("TAVILY_API_KEY not found.") 
        return []

    tool = TavilySearch(max_results=max_results)

    try:
        response = tool.invoke(query)
        
        print("========== RAW TAVILY RESPONSE ==========")
        print(response)
        print("=========================================")

        if isinstance(response , list):
            return response 
        
        if isinstance(response , dict):
            return response.get("results" , []) 
        
        return []

        
    except Exception as e:
        print("Tavily Error:", e)
        return []



def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))
        
        print("=" * 60)
        print("RAW RESULTS")
        print(raw)
        print("TOTAL:", len(raw))
        print("=" * 60)

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    _log("worker", f"Starting task {task.id}: {task.title}")

    start_time = time.time()

    bullets_text = "\n".join(f"- {b}" for b in task.bullets)

    constraints_text = ", ".join(plan.constraints) if plan.constraints else "None"

    if task.requires_citations and evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'Unknown'}"
            for e in evidence[:5]
        )
    else:
        evidence_text = "No evidence required."

    response = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=f"""
Blog Title: {plan.blog_title}

Audience: {plan.audience}

Tone: {plan.tone}

Blog Type: {plan.blog_kind}

Constraints: {constraints_text}

Topic: {payload['topic']}

Mode: {payload.get('mode')}

As-of Date: {payload.get('as_of')}

Section Title: {task.title}

Goal:
{task.goal}

Target Words:
{task.target_words}

Tags:
{", ".join(task.tags) if task.tags else "None"}

Requires Research:
{task.requires_research}

Requires Citations:
{task.requires_citations}

Requires Code:
{task.requires_code}

Section Outline:
{bullets_text}

Evidence:
{evidence_text}
"""
            ),
        ]
    )

    section_md = response.content.strip()

    elapsed = time.time() - start_time

    _log(
        "worker",
        f"Finished task {task.id} in {elapsed:.1f}s"
    )

    return {
        "sections": [
            (task.id, section_md)
        ]
    }

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    """
    Runs only if the user enabled image generation.
    Otherwise simply passes the merged markdown through unchanged.
    """

    # User disabled images
    if not state.get("include_images", False):
        return {
            "md_with_placeholders": state["merged_md"],
            "image_specs": [],
        }

    planner = llm.with_structured_output(GlobalImagePlan)

    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders where useful.\n"
                    "Maximum 3 diagrams.\n\n"
                    f"{state['merged_md']}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [
            img.model_dump()
            for img in image_plan.images
        ],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Depending on SDK version, parts may hang off resp.candidates[0].content.parts  
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    # ----------------------------------------------------
    # Output folders
    # ----------------------------------------------------
    blogs_dir = Path("generated_blogs")
    blogs_dir.mkdir(exist_ok=True)

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    blog_path = blogs_dir / f"{_safe_slug(plan.blog_title)}.md"

    # ----------------------------------------------------
    # Images disabled from UI
    # ----------------------------------------------------
    if not state.get("include_images", False):
        md = state["merged_md"]

        blog_path.write_text(
            md,
            encoding="utf-8"
        )

        return {
            "final": md,
            "image_results": []
        }

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    image_results = []

    # ----------------------------------------------------
    # Planner decided no images are required
    # ----------------------------------------------------
    if not image_specs:

        blog_path.write_text(
            md,
            encoding="utf-8"
        )

        return {
            "final": md,
            "image_results": image_results
        }

    # ----------------------------------------------------
    # Generate each image
    # ----------------------------------------------------
    for spec in image_specs:

        placeholder = spec["placeholder"]
        filename = spec["filename"]

        image_path = images_dir / filename

        try:

            if not image_path.exists():

                image_bytes = _gemini_generate_image_bytes(
                    spec["prompt"]
                )

                image_path.write_bytes(image_bytes)

            markdown_image = (
                f'![{spec["alt"]}](images/{filename})\n\n'
                f'*{spec["caption"]}*'
            )

            md = md.replace(
                placeholder,
                markdown_image
            )

            image_results.append(
                {
                    "status": "success",
                    "filename": filename,
                    "caption": spec["caption"],
                    "alt": spec["alt"],
                    "path": str(image_path),
                }
            )

        except Exception as e:

            image_results.append(
                {
                    "status": "failed",
                    "filename": filename,
                    "caption": spec["caption"],
                    "alt": spec["alt"],
                    "prompt": spec["prompt"],
                    "reason": str(e),
                }
            )

            # Remove placeholder from the blog
            md = md.replace(
                placeholder,
                ""
            )

    # ----------------------------------------------------
    # Save final blog
    # ----------------------------------------------------
    blog_path.write_text(
        md,
        encoding="utf-8"
    )

    return {
        "final": md,
        "image_results": image_results,
    }

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app

