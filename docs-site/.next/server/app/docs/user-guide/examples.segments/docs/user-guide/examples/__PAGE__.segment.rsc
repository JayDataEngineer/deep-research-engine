1:"$Sreact.fragment"
35:I[82276,["/_next/static/chunks/0ypdq~u812b.d.js","/_next/static/chunks/0vup3~.t2.xes.js"],"OutletBoundary"]
36:"$Sreact.suspense"
0:{"rsc":["$","$1","c",{"children":[[["$","header",null,{"className":"mb-24 mt-12 bg-surface p-0","children":[["$","div",null,{"className":"w-12 h-1 bg-primary-container mb-8"}],["$","h1",null,{"className":"text-[3.5rem] font-headline font-bold leading-none tracking-tight text-on-surface","children":"Examples"}]]}],"\n",["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Practical examples of using the Deep Research Engine."}],"\n",["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"RAG Research Example"]}],"\n",["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Research a topic using web search and vector storage."}],"\n",["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Configuration"]}],"\n",["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-yaml font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"# configs/rag.yaml\nworkflow:\n  name: rag_research\n  type: research\n  version: 1.0.0\n\ninput:\n  type: query\n  source: \"${QUERY}\"\n\nphases:\n  - name: ingestion\n    node: IngestionNode\n\n  - name: drafter\n    node: DrafterNode\n    config:\n      strategy: complexity_based\n\n  - name: critic\n    node: CriticNode\n\n  - name: synthesis\n    node: SynthesisNode\n    config:\n      format: research_report\n      citations:\n        enabled: true\n\nstorage:\n  postgres:\n    enabled: true\n  memory:\n    enabled: true\n\nllm:\n  models:\n    main: qwen-35-27\n    tool: qwen-35-27-fast\n"}]}]]}]]}],"\n",["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Run with CLI"]}],"\n",["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-bash font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"deep-research research \"What is quantum computing?\" --config configs/rag.yaml\n"}]}]]}]]}],"\n","$L2","\n","$L3","\n","$L4","\n","$L5","\n","$L6","\n","$L7","\n","$L8","\n","$L9","\n","$La","\n","$Lb","\n","$Lc","\n","$Ld","\n","$Le","\n","$Lf","\n","$L10","\n","$L11","\n","$L12","\n","$L13","\n","$L14","\n","$L15","\n","$L16","\n","$L17","\n","$L18","\n","$L19","\n","$L1a","\n","$L1b","\n","$L1c","\n","$L1d","\n","$L1e","\n","$L1f","\n","$L20","\n","$L21","\n","$L22","\n","$L23","\n","$L24","\n","$L25","\n","$L26","\n","$L27","\n","$L28","\n","$L29","\n","$L2a","\n","$L2b","\n","$L2c","\n","$L2d","\n","$L2e","\n","$L2f"],null,"$L30"]}],"isPartial":false,"staleTime":300,"varyParams":null,"buildId":"fGZt_DnWuGacpFmz7jr05"}
2:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Run with Python"]}]
3:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-python font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"from deep_research_engine import run_workflow, load_config\nimport asyncio\n\nasync def main():\n    config = load_config(\"configs/rag.yaml\")\n    \n    async for event in run_workflow(\"What is quantum computing?\", config):\n        if event[\"type\"] == \"phase_complete\":\n            print(f\"✓ Phase {event['phase']} complete\")\n        elif event[\"type\"] == \"synthesis_complete\":\n            print(f\"\\nResult:\\n{event['output']}\")\n\nasyncio.run(main())\n"}]}]]}]]}]
4:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
5:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"PDF Summarizer Example"]}]
6:["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Process and summarize a PDF document."}]
7:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Configuration"]}]
8:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-yaml font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"# configs/pdf_summarizer.yaml\nworkflow:\n  name: pdf_summarizer\n  type: pdf_summarization\n  version: 1.0.0\n\ninput:\n  type: pdf\n  source: \"${PDF_PATH}\"\n\nphases:\n  - name: ingestion\n    node: IngestionNode\n\n  - name: drafter\n    node: DrafterNode\n    config:\n      strategy: simple\n\n  - name: critic\n    node: CriticNode\n\n  - name: synthesis\n    node: SynthesisNode\n    config:\n      format: summary\n\nstorage:\n  memory:\n    enabled: true\n\nexecution:\n  max_steps: 50\n  timeout: 300\n"}]}]]}]]}]
9:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Run with CLI"]}]
a:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-bash font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"deep-research pdf ./book.pdf --config configs/pdf_summarizer.yaml\n"}]}]]}]]}]
b:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Run with Python"]}]
c:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-python font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"from deep_research_engine import run_workflow, load_config\nimport asyncio\n\nasync def main():\n    config = load_config(\"configs/pdf_summarizer.yaml\")\n    \n    async for event in run_workflow(\"\", config, pdf_path=\"./book.pdf\"):\n        if event[\"type\"] == \"synthesis_complete\":\n            print(f\"\\nSummary:\\n{event['output']}\")\n\nasyncio.run(main())\n"}]}]]}]]}]
d:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
e:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"Telegram Ingestion Example"]}]
f:["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Process Telegram chat exports with multimodal support."}]
10:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Configuration"]}]
11:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-yaml font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"# configs/telegram_vibevoice.yaml\nworkflow:\n  name: telegram_ingest\n  type: telegram_ingest\n  version: 1.0.0\n\ninput:\n  type: telegram_export\n  source: \"${TELEGRAM_EXPORT_PATH}\"\n\nphases:\n  - name: ingestion\n    node: IngestionNode\n    config:\n      extract_audio: true\n      extract_frames: true\n      asr_model: vibevoice\n      diarization: true\n\n  - name: drafter\n    node: DrafterNode\n\n  - name: critic\n    node: CriticNode\n\n  - name: neo4j\n    node: Neo4jBuilderNode\n\n  - name: synthesis\n    node: SynthesisNode\n\nstorage:\n  postgres:\n    enabled: true\n  neo4j:\n    enabled: true\n  mongodb:\n    enabled: true\n  minio:\n    enabled: true\n"}]}]]}]]}]
12:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Run with CLI"]}]
13:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-bash font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"deep-research ingest ./ChatExport_2026-03-13 --config configs/telegram_vibevoice.yaml\n"}]}]]}]]}]
14:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
15:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"Skill Generator Example"]}]
16:["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Scrape documentation and generate skills."}]
17:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Configuration"]}]
18:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-yaml font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"# configs/skill_generator.yaml\nworkflow:\n  name: skill_generator\n  type: skill_generation\n  version: 1.0.0\n\ninput:\n  type: urls\n  source:\n    - https://docs.python.org\n    - https://docs.djangoproject.com\n\nphases:\n  - name: ingestion\n    node: IngestionNode\n\n  - name: drafter\n    node: DrafterNode\n    config:\n      worker_type: scraper\n\n  - name: critic\n    node: CriticNode\n\n  - name: synthesis\n    node: SynthesisNode\n    config:\n      format: skill_cards\n\nstorage:\n  memory:\n    enabled: true\n"}]}]]}]]}]
19:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Run with CLI"]}]
1a:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-bash font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"deep-research scrape https://docs.example.com --config configs/skill_generator.yaml\n"}]}]]}]]}]
1b:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
1c:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"FastAPI Integration"]}]
1d:["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Integrate the engine into a FastAPI application."}]
31:T406,from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from deep_research_engine import run_workflow, load_config
import asyncio
import json

app = FastAPI()

@app.post("/api/research")
async def research(query: str):
    """Start research and stream results."""
    config = load_config("configs/rag.yaml")

    async def generate():
        async for event in run_workflow(query, config, stream=True):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/research/async")
async def research_async(query: str, background_tasks: BackgroundTasks):
    """Start research in background."""
    config = load_config("configs/rag.yaml")
    
    async def run_research():
        async for event in run_workflow(query, config):
            # Store results or send notifications
            pass
    
    background_tasks.add_task(asyncio.run, run_research())
    return {"status": "started", "query": query}
1e:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-python font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"$31"}]}]]}]]}]
1f:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
20:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"Custom Worker Example"]}]
21:["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Create and use a custom worker."}]
32:T52d,from deep_research_engine.workers import BaseWorker, WorkerConfig, WorkerRegistry
from deep_research_engine.state import EngineState, WorkerResult
import httpx

@WorkerRegistry.register("api_call")
class APICallWorker(BaseWorker):
    """Worker for making API calls."""

    worker_type = "api_call"
    description = "Makes HTTP requests to external APIs"

    async def execute(self, state: EngineState, config: WorkerConfig) -> WorkerResult:
        """Execute API call."""
        self._start_timing()

        try:
            url = config.custom.get("url")
            method = config.custom.get("method", "GET")
            headers = config.custom.get("headers", {})

            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, headers=headers)
                response.raise_for_status()

                return self._create_result(
                    success=True,
                    output=response.json(),
                    output_type="api_response",
                    confidence=0.9,
                    sources=[{"type": "api", "url": url}]
                )

        except Exception as e:
            return self._create_result(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
22:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-python font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"$32"}]}]]}]]}]
23:["$","h3",null,{"className":"text-xl font-headline font-bold mb-6 mt-12 text-on-surface uppercase tracking-tight flex items-center gap-3","children":[["$","span",null,{"className":"w-6 h-0.5 bg-primary-container"}],"Use Custom Worker in Config"]}]
24:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-yaml font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"phases:\n  - name: workers\n    node: WorkerNode\n    config:\n      worker_type: api_call\n      custom:\n        url: \"https://api.example.com/data\"\n        method: \"GET\"\n"}]}]]}]]}]
25:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
26:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"Multi-Workflow Orchestration"]}]
27:["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Run multiple workflows in sequence."}]
33:T41c,from deep_research_engine import run_workflow, load_config
import asyncio

async def run_research_pipeline(query: str):
    """Run multi-stage research pipeline."""
    
    # Stage 1: Initial research
    rag_config = load_config("configs/rag.yaml")
    initial_findings = []
    
    async for event in run_workflow(query, rag_config):
        if event["type"] == "synthesis_complete":
            initial_findings = event["output"]
    
    # Stage 2: Deep dive
    pdf_config = load_config("configs/pdf_summarizer.yaml")
    summaries = []
    
    for pdf_path in ["paper1.pdf", "paper2.pdf"]:
        async for event in run_workflow("", pdf_config, pdf_path=pdf_path):
            if event["type"] == "synthesis_complete":
                summaries.append(event["output"])
    
    # Stage 3: Final synthesis
    return {
        "initial_findings": initial_findings,
        "summaries": summaries,
        "combined": initial_findings + summaries
    }

# Run pipeline
results = asyncio.run(run_research_pipeline("Quantum computing advances"))
28:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-python font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"$33"}]}]]}]]}]
29:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
2a:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"Streaming Events"]}]
2b:["$","p",null,{"className":"mb-8 leading-relaxed text-on-surface-variant font-body text-lg max-w-3xl","children":"Handle streaming events from the engine."}]
34:T41b,from deep_research_engine import run_workflow, load_config

async def stream_research(query: str):
    config = load_config("configs/rag.yaml")
    
    async for event in run_workflow(query, config, stream=True):
        event_type = event.get("type")
        
        if event_type == "phase_start":
            print(f"🔄 Starting phase: {event['phase']}")
        
        elif event_type == "phase_complete":
            print(f"✓ Completed phase: {event['phase']}")
        
        elif event_type == "worker_complete":
            print(f"  → Worker {event['worker_id']} complete")
        
        elif event_type == "critic_decision":
            if event["decision"] == "pass":
                print(f"✓ Critic approved")
            else:
                print(f"↺ Critic requested revision: {event['feedback']}")
        
        elif event_type == "synthesis_complete":
            print(f"\n📄 Final Report:\n{event['output']}")
        
        elif event_type == "error":
            print(f"❌ Error: {event['error']}")
2c:["$","div",null,{"className":"relative mb-16 group shadow-2xl","children":[["$","div",null,{"className":"absolute top-0 right-0 px-4 py-2 bg-primary text-on-primary text-[9px] font-bold font-headline uppercase tracking-[0.25em] z-10","children":"TERMINAL_S01 // SRC_LOG"}],["$","pre",null,{"className":"p-10 bg-[#0a0c0c] text-[#00e5ff] overflow-x-auto border-l-4 border-primary/30","children":[["$","div",null,{"className":"flex items-center gap-2 mb-10 opacity-20","children":[["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}],["$","div",null,{"className":"w-1.5 h-1.5 bg-[#00e5ff] rounded-full"}]]}],["$","div",null,{"className":"font-mono text-[13px] selection:bg-primary/30","children":["$","code",null,{"className":"language-python font-mono text-[0.92em] font-bold tracking-tight text-[#00e5ff] font-medium block leading-relaxed","style":{"color":"#00e5ff"},"children":"$34"}]}]]}]]}]
2d:["$","div",null,{"className":"my-24 flex items-center gap-4","children":[["$","hr",null,{"className":"flex-1 border-surface-container"}],["$","span",null,{"className":"material-symbols-outlined text-outline-variant text-sm","children":"architecture"}],["$","hr",null,{"className":"flex-1 border-surface-container"}]]}]
2e:["$","h2",null,{"className":"text-2xl font-headline font-bold mb-10 mt-20 text-on-surface uppercase tracking-tighter flex items-center gap-4","children":[["$","span",null,{"className":"w-2 h-8 bg-primary"}],"Reference"]}]
2f:["$","ul",null,{"className":"mb-10 space-y-4 text-on-surface-variant font-body list-none","children":["\n",["$","li",null,{"className":"flex items-start gap-4","children":[["$","span",null,{"className":"material-symbols-outlined text-primary text-sm mt-1","children":"arrow_forward"}],["$","span",null,{"children":["$","a",null,{"href":"../workflows","children":"Workflow Configs"}]}]]}],"\n",["$","li",null,{"className":"flex items-start gap-4","children":[["$","span",null,{"className":"material-symbols-outlined text-primary text-sm mt-1","children":"arrow_forward"}],["$","span",null,{"children":["$","a",null,{"href":"../workers","children":"Workers Guide"}]}]]}],"\n",["$","li",null,{"className":"flex items-start gap-4","children":[["$","span",null,{"className":"material-symbols-outlined text-primary text-sm mt-1","children":"arrow_forward"}],["$","span",null,{"children":["$","a",null,{"href":"../storage","children":"Storage Backends"}]}]]}],"\n",["$","li",null,{"className":"flex items-start gap-4","children":[["$","span",null,{"className":"material-symbols-outlined text-primary text-sm mt-1","children":"arrow_forward"}],["$","span",null,{"children":["$","a",null,{"href":"../../api-reference","children":"API Reference"}]}]]}],"\n"]}]
30:["$","$L35",null,{"children":["$","$36",null,{"name":"Next.MetadataOutlet","children":"$@37"}]}]
37:null
