# Graph Builder Node Plan

## Overview

A **SINGLE ISOLATED NODE** that transforms research findings into a structured knowledge graph. This runs **once, right before synthesis**. All Neo4j creation happens HERE - Flash Explorers NO LONGER write to Neo4j.

## Why Embeddings Come After Entity Extraction

```
Text Chunk → LLMGraphTransformer → Entity (id + description)
                                              ↓
                                    NOW embed the entity
                                    (entity.id + " " + entity.description)
```

**You can't embed something that doesn't exist yet.** The LLM creates entity objects from text, THEN you create embeddings for those entities. Same for communities - you detect communities, summarize them, THEN embed the summaries.

## Architecture

```
Boss → Lab Architect → Flash Explorers
                              ↓
                        Context Engine
                        (Postgres + MinIO ONLY)
                        (STOP writing to Neo4j!)
                              ↓
                    ┌─────────────────────────┐
                    │    GRAPH BUILDER NODE   │
                    │    (SINGLE EXECUTION)   │
                    │    (ISOLATED)           │
                    │                         │
                    │  1. Pull findings from  │
                    │     Context Engine      │
                    │  2. Create entity       │
                    │     chunks from findings│
                    │  3. LLMGraphTransformer │
                    │     (concepts + entities)│
                    │  4. Schema consolidation│
                    │     (LLM normalizes)    │
                    │  5. Build Neo4j graph   │
                    │  6. Entity embeddings   │
                    │  7. Leiden communities  │
                    │  8. Community summaries │
                    │  9. Community embeddings│
                    └─────────────────────────┘
                              ↓
                         Neo4j Graph
                              ↓
         ┌────────────────────┴────────────────────┐
         ↓                                         ↓
    Synthesis                                  Critic
    (graph query tools)                        (graph query tools)
```

## Node Design: GraphBuilderNode

### Location
```
src/extensions/content/deep_research/
├── nodes/
│   └── graph_builder/
│       ├── __init__.py
│       ├── node.py                 # Main node - orchestrates everything
│       ├── chunker.py              # Convert findings → entity chunks
│       ├── extractor.py            # LLMGraphTransformer (concepts!)
│       ├── consolidator.py         # Schema normalization via LLM
│       ├── builder.py              # Neo4j graph construction
│       ├── embedder.py             # Entity + community embeddings
│       ├── community_detector.py   # Leiden algorithm
│       └── config.yaml
│
├── tools/                          # TOOLS FOR SYNTHESIS/CRITIC
│   ├── graph_search.py             # Vector search on entities
│   ├── community_search.py         # Search community summaries
│   ├── graph_traverse.py           # Multi-hop graph traversal
│   └── natural_language_query.py   # Text → Cypher via LLM
```

---

## STEP 1: Input - Where Data Comes From

The Graph Builder pulls from **Context Engine** (NOT from Flash Explorers directly):

```python
# chunker.py

from src.extensions.content.deep_research.context_engine import ContextEngine

class FindingChunker:
    """Convert research findings into entity-extraction-ready chunks"""

    def __init__(self, context: ExtensionContext):
        self.context_engine = ContextEngine(context)

    async def get_findings_as_chunks(self, session_id: str) -> List[EntityChunk]:
        """
        Pull findings from Context Engine and format for entity extraction.

        Sources:
        - Postgres: FindingArtifact, ClaimArtifact, CitationArtifact
        - MinIO: Raw text files (>5000 chars)
        """

        # Get all artifacts for this session
        artifacts = await self.context_engine.get_session_artifacts(session_id)

        chunks = []
        for artifact in artifacts:
            if artifact.type == "FindingArtifact":
                chunk = EntityChunk(
                    id=f"finding_{artifact.id}",
                    text=artifact.content,  # The actual finding text
                    metadata={
                        "source_url": artifact.source_url,
                        "source_type": artifact.source_type,
                        "confidence": artifact.confidence,
                        "worker_id": artifact.worker_id,
                        "artifact_type": "finding"
                    }
                )
            elif artifact.type == "ClaimArtifact":
                chunk = EntityChunk(
                    id=f"claim_{artifact.id}",
                    text=f"{artifact.claim}\nEvidence: {artifact.evidence}",
                    metadata={
                        "source_url": artifact.source_url,
                        "artifact_type": "claim"
                    }
                )
            # ... other artifact types

            chunks.append(chunk)

        return chunks

class EntityChunk(TypedDict):
    """A chunk of text ready for entity extraction"""
    id: str
    text: str
    metadata: dict
```

---

## STEP 2: Entity/Concept Extraction

```python
# extractor.py

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# FOCUSED ON CONCEPTS, NOT JUST NAMED ENTITIES
ALLOWED_NODE_TYPES = [
    "Concept",      # Ideas, theories, frameworks
    "Finding",      # Specific claims, results
    "Method",       # Methodologies, approaches
    "Tool",         # Technologies, software
    "Person",
    "Organization",
    "Dataset",
    "Topic",
]

ALLOWED_RELATIONSHIPS = [
    # Concept relationships
    "RELATES_TO",
    "SUPPORTS",        # A provides evidence for B
    "CONTRADICTS",     # A conflicts with B
    "DERIVED_FROM",    # A builds on B
    "SUBSUMES",        # A contains B
    "APPLIES",         # A uses B

    # Attribution
    "ATTRIBUTED_TO",   # Finding → Person/Org
    "FOUND_BY",        # Finding → Method
    "MEASURED_IN",     # Finding → Dataset
]

class ConceptExtractor:
    """Extract concepts and entities using LLMGraphTransformer"""

    def __init__(self, context: ExtensionContext):
        self.llm = context.get_llm()
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=ALLOWED_NODE_TYPES,
            allowed_relationships=ALLOWED_RELATIONSHIPS,
            node_properties=["description", "confidence"],
            strict_mode=False
        )

    async def extract(self, chunks: List[EntityChunk]) -> List[GraphDocument]:
        """
        Extract entities and relationships from chunks.

        Input: List of EntityChunk (text from findings)
        Output: List of GraphDocument (nodes + edges)
        """

        graph_documents = []

        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )

            # LLM extracts entities/concepts from this text
            graph_doc = await self.transformer.aconvert_to_graph_documents([doc])

            # Tag with source chunk for provenance
            for node in graph_doc[0].nodes:
                node.properties["chunk_id"] = chunk["id"]
                node.properties["source_url"] = chunk["metadata"].get("source_url")

            graph_documents.extend(graph_doc)

        return graph_documents
```

### Main Node Class

```python
# node.py

from langgraph.graph import StateGraph, END
from typing import Literal
from src.extensions.modes.brain_modes.deep_agent import DeepAgentNode

class GraphBuilderNode:
    """
    Deep research graph builder node.

    Transforms raw findings into a structured knowledge graph
    with concept-focused communities for synthesis.
    """

    def __init__(self, context: ExtensionContext):
        self.context = context
        self.neo4j_uri = context.config.get("neo4j_uri", "bolt://localhost:7687")
        self.neo4j_user = context.config.get("neo4j_user", "neo4j")
        self.neo4j_password = context.config.get("neo4j_password")

        # Sub-components
        self.extractor = ConceptExtractor(context)
        self.consolidator = SchemaConsolidator(context)
        self.builder = GraphBuilder(context)
        self.community_detector = CommunityDetector(context)

    async def __call__(self, state: GraphBuilderState) -> GraphBuilderState:
        """Execute the graph building pipeline"""

        try:
            state["status"] = "extracting"

            # Step 1: Extract chunks from findings
            chunks = await self._prepare_chunks(state["findings"])
            state["chunks_processed"] = len(chunks)

            # Step 2: Extract entities and concepts
            graph_documents = await self.extractor.extract(chunks)
            state["entities_extracted"] = sum(len(g.nodes) for g in graph_documents)

            # Step 3: Consolidate schema (normalize entity types)
            consolidated = await self.consolidator.consolidate(graph_documents)

            # Step 4: Build graph in Neo4j
            await self.builder.build_graph(
                consolidated,
                session_id=state["session_id"]
            )
            state["relationships_created"] = consolidated.relationship_count

            # Step 5: Detect communities
            communities = await self.community_detector.detect_communities(
                session_id=state["session_id"]
            )
            state["communities_created"] = len(communities)
            state["community_summaries"] = communities

            # Step 6: Create embeddings for retrieval
            await self.community_detector.create_embeddings(
                session_id=state["session_id"]
            )

            state["graph_ready"] = True
            state["status"] = "completed"

            return state

        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)
            return state

    async def _prepare_chunks(self, findings: List[FindingArtifact]) -> List[Chunk]:
        """Convert findings into processable chunks"""
        chunks = []
        for i, finding in enumerate(findings):
            chunk = Chunk(
                id=f"{finding['finding_id']}_chunk_{i}",
                text=finding["content"],
                metadata={
                    "source_url": finding["source_url"],
                    "source_type": finding["source_type"],
                    "worker_id": finding["worker_id"],
                    "confidence": finding["confidence"],
                    "session_id": finding.get("session_id")
                }
            )
            chunks.append(chunk)
        return chunks
```

---

## STEP 3: Schema Consolidation

LLMs produce inconsistent entity types ("Person", "PERSON", "people", "Human"). Clean them up.

```python
# consolidator.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

CONSOLIDATION_PROMPT = """
Normalize these entity types and relationships to canonical forms.

Canonical Entity Types:
- Concept: Ideas, theories, frameworks, paradigms
- Finding: Specific claims, results, discoveries
- Method: Methodologies, approaches, techniques
- Tool: Technologies, software, instruments
- Person: People, researchers, authors
- Organization: Companies, institutions, groups
- Dataset: Data sources, corpora
- Topic: High-level subject areas

Canonical Relationships:
- RELATES_TO, SUPPORTS, CONTRADICTS, DERIVED_FROM, APPLIES
- FOUND_BY, ATTRIBUTED_TO, SUBSUMES

Input nodes: {nodes}
Input relationships: {relationships}

Output JSON mapping variants to canonical forms.
"""

class SchemaConsolidator:
    """LLM normalizes inconsistent entity types"""

    def __init__(self, context: ExtensionContext):
        self.llm = context.get_llm()

    async def consolidate(self, graph_documents: List[GraphDocument]) -> List[GraphDocument]:
        # Collect all types
        node_types = set(n.type for doc in graph_documents for n in doc.nodes)
        rel_types = set(r.type for doc in graph_documents for r in doc.relationships)

        # Get mappings from LLM
        prompt = ChatPromptTemplate.from_template(CONSOLIDATION_PROMPT)
        chain = prompt | self.llm | JsonOutputParser()
        mappings = await chain.ainvoke({"nodes": list(node_types), "relationships": list(rel_types)})

        # Apply mappings
        for doc in graph_documents:
            for node in doc.nodes:
                node.type = self._map_node_type(node.type, mappings["nodes"])
            for rel in doc.relationships:
                rel.type = self._map_rel_type(rel.type, mappings["relationships"])

        return graph_documents
```

---

## STEP 4: Build Neo4j Graph

Store consolidated entities with session isolation.

```python
# builder.py

class Neo4jGraphBuilder:
    """Build the knowledge graph in Neo4j"""

    async def build_graph(self, graph_documents: List[GraphDocument], session_id: str):
        async with self.driver.session() as session:
            # 1. Create Document node
            await session.run("""
                MERGE (d:Document {session_id: $session_id})
                SET d.created_at = datetime()
            """, session_id=session_id)

            # 2. Create Chunk nodes (provenance)
            await self._create_chunks(session, graph_documents, session_id)

            # 3. Create Entity nodes
            await self._create_entities(session, graph_documents, session_id)

            # 4. Create Entity relationships
            await self._create_relationships(session, graph_documents, session_id)

    async def _create_entities(self, session, graph_documents, session_id):
        """Create entities with MENTIONED_IN → chunks"""
        entities = []
        for doc in graph_documents:
            for node in doc.nodes:
                entities.append({
                    "id": node.id,
                    "type": node.type,
                    "description": node.properties.get("description", ""),
                    "chunk_id": node.properties.get("chunk_id"),
                    "session_id": session_id
                })

        await session.run("""
            UNWIND $entities AS e
            MERGE (entity:Entity {id: e.id})
            SET entity.type = e.type,
                entity.description = e.description,
                entity.session_ids = coalesce(entity.session_ids, []) +
                    CASE WHEN e.session_id IN entity.session_ids THEN [] ELSE [e.session_id] END
            WITH entity, e
            MATCH (chunk:Chunk {id: e.chunk_id})
            MERGE (chunk)-[:MENTIONED_IN]->(entity)
        """, entities=entities)
```

---

## STEP 5: Entity Embeddings

**NOW we embed the entities.** We couldn't do this before because entities didn't exist.

```python
# embedder.py

class EntityEmbedder:
    """Create embeddings for entities"""

    def __init__(self, context: ExtensionContext):
        self.embeddings = context.get_embeddings()
        self.driver = ...

    async def embed_entities(self, session_id: str):
        """
        Embed each entity: (id + " " + description) → vector

        This enables similarity search on entities.
        """
        async with self.driver.session() as session:
            # Get entities without embeddings
            result = await session.run("""
                MATCH (e:Entity)
                WHERE $session_id IN e.session_ids
                AND e.embedding IS NULL
                RETURN e.id as id, e.description as description
            """, session_id=session_id)

            entities = [record async for record in result]

            for entity in entities:
                # Create text to embed
                text = f"{entity['id']} {entity['description']}"

                # Generate embedding
                embedding = await self.embeddings.aembed_query(text)

                # Store in Neo4j
                await session.run("""
                    MATCH (e:Entity {id: $id})
                    CALL db.create.setNodeVectorProperty(e, "embedding", $embedding)
                """, id=entity['id'], embedding=embedding)
```

---

## STEP 6: Community Detection (Leiden)

Now detect clusters of related concepts using GraphDataScience.

```python
# community_detector.py

from graphdatascience import GraphDataScience

class CommunityDetector:
    """Detect concept communities using Leiden algorithm"""

    def __init__(self, context: ExtensionContext):
        self.gds = GraphDataScience(
            context.config["neo4j_uri"],
            auth=(context.config["neo4j_user"], context.config["neo4j_password"])
        )

    async def detect_communities(self, session_id: str, max_levels: int = 3):
        """
        1. Project entity graph (entities + their relationships)
        2. Run Leiden algorithm
        3. Create Community nodes with IN_COMMUNITY relationships
        """

        # 1. Project graph (only entities, not chunks)
        projection_name = f"communities_{session_id}"

        self.gds.run_cypher("""
            MATCH (e:Entity)-[r]->(e2:Entity)
            WHERE $session_id IN e.session_ids
            AND $session_id IN e2.session_ids
            WITH e, e2, count(r) as weight
            CALL gds.graph.project(
                $name, e, e2,
                { relationshipProperties: { weight: weight } },
                { undirectedRelationshipTypes: ['*'] }
            )
        """, {"session_id": session_id, "name": projection_name})

        # 2. Run Leiden
        self.gds.leiden.write(
            self.gds.graph.get(projection_name),
            writeProperty="communities",
            includeIntermediateCommunities=True,
            maxLevels=max_levels
        )

        # 3. Create Community nodes
        await self._create_community_nodes(session_id)

        # Cleanup
        self.gds.graph.drop(projection_name)

    async def _create_community_nodes(self, session_id: str):
        """Create hierarchical Community nodes"""

        # Level 0 communities
        self.gds.run_cypher("""
            MATCH (e:Entity)
            WHERE e.communities IS NOT NULL
            AND $session_id IN e.session_ids
            WITH e, e.communities[0] AS cid
            MERGE (c:Community {id: '0-' + toString(cid), session_id: $session_id})
            ON CREATE SET c.level = 0
            MERGE (e)-[:IN_COMMUNITY]->(c)
        """, {"session_id": session_id})

        # Higher levels with PARENT_COMMUNITY relationships
        self.gds.run_cypher("""
            MATCH (e:Entity)
            WHERE e.communities IS NOT NULL AND size(e.communities) > 1
            AND $session_id IN e.session_ids
            UNWIND range(1, size(e.communities) - 1) AS level
            MERGE (curr:Community {id: toString(level) + '-' + toString(e.communities[level]), session_id: $session_id})
            ON CREATE SET curr.level = level
            MERGE (prev:Community {id: toString(level-1) + '-' + toString(e.communities[level-1]), session_id: $session_id})
            MERGE (prev)-[:PARENT_COMMUNITY]->(curr)
        """, {"session_id": session_id})
```

---

## STEP 7: Community Summaries + Embeddings

LLM summarizes each community, then we embed those summaries.

```python
class CommunitySummarizer:
    """LLM generates summaries for each community"""

    async def summarize_communities(self, session_id: str) -> List[CommunitySummary]:
        # Get communities with their entities
        communities = self.gds.run_cypher("""
            MATCH (c:Community {session_id: $session_id})<-[:IN_COMMUNITY]-(e:Entity)
            WITH c, collect({type: e.type, id: e.id, desc: e.description}) AS entities
            RETURN c.id AS id, c.level AS level, entities
        """, {"session_id": session_id})

        summaries = []
        for _, row in communities.iterrows():
            entity_list = "\n".join([f"- {e['type']}: {e['id']}" for e in row['entities']])

            prompt = f"""These concepts/entities are related in research:

{entity_list}

Generate:
title: [4 word title]
summary: [2-3 sentences about what connects them]"""

            response = await self.llm.ainvoke(prompt)
            title, summary = self._parse_response(response.content)

            summaries.append({
                "id": row['id'],
                "level": row['level'],
                "title": title,
                "summary": summary
            })

        # Store in Neo4j
        self.gds.run_cypher("""
            UNWIND $summaries AS s
            MATCH (c:Community {id: s.id, session_id: $session_id})
            SET c.title = s.title, c.summary = s.summary
        """, {"session_id": session_id, "summaries": summaries})

        return summaries

    async def embed_communities(self, session_id: str):
        """Embed community summaries for similarity search"""
        communities = self.gds.run_cypher("""
            MATCH (c:Community {session_id: $session_id})
            WHERE c.summary IS NOT NULL
            RETURN c.id AS id, c.summary AS text
        """, {"session_id": session_id})

        for _, row in communities.iterrows():
            embedding = await self.embeddings.aembed_query(row['text'])
            self.gds.run_cypher("""
                MATCH (c:Community {id: $id})
                CALL db.create.setNodeVectorProperty(c, "embedding", $embedding)
            """, id=row['id'], embedding=embedding)
```

---

## MAIN NODE: GraphBuilderNode

```python
# node.py

class GraphBuilderNode:
    """
    Single isolated node that builds the entire knowledge graph.

    Runs ONCE before synthesis.
    All Neo4j creation happens HERE - nowhere else.
    """

    def __init__(self, context: ExtensionContext):
        self.chunker = FindingChunker(context)
        self.extractor = ConceptExtractor(context)
        self.consolidator = SchemaConsolidator(context)
        self.builder = Neo4jGraphBuilder(context)
        self.embedder = EntityEmbedder(context)
        self.community_detector = CommunityDetector(context)
        self.community_summarizer = CommunitySummarizer(context)

    async def __call__(self, state: GlobalState) -> GlobalState:
        session_id = state["session_id"]

        try:
            # STEP 1: Get findings from Context Engine
            chunks = await self.chunker.get_findings_as_chunks(session_id)
            logger.info(f"Got {len(chunks)} chunks from Context Engine")

            # STEP 2: Extract entities/concepts
            graph_docs = await self.extractor.extract(chunks)
            logger.info(f"Extracted {sum(len(d.nodes) for d in graph_docs)} entities")

            # STEP 3: Consolidate schema
            graph_docs = await self.consolidator.consolidate(graph_docs)

            # STEP 4: Build Neo4j graph
            await self.builder.build_graph(graph_docs, session_id)

            # STEP 5: Embed entities
            await self.embedder.embed_entities(session_id)

            # STEP 6: Detect communities
            await self.community_detector.detect_communities(session_id)

            # STEP 7: Summarize + embed communities
            summaries = await self.community_summarizer.summarize_communities(session_id)
            await self.community_summarizer.embed_communities(session_id)

            logger.info(f"Graph built with {len(summaries)} communities")

            return {
                **state,
                "graph_built": True,
                "community_summaries": summaries
            }

        except Exception as e:
            logger.error(f"Graph builder failed: {e}")
            return {**state, "graph_built": False, "graph_error": str(e)}
```

---

## INTEGRATION: Update Research Graph

```python
# In your main graph.py

# Flash Explorers NO LONGER write to Neo4j
# They only write to Postgres/MinIO

# Add GraphBuilderNode BEFORE synthesis
graph.add_node("graph_builder", GraphBuilderNode(context))

# Routing
# Boss → Lab Architect → Flash Explorers → graph_builder → Synthesis → Critic
```

---

## TOOLS FOR SYNTHESIS/CRITIC

Now give synthesis and critic nodes tools to query the graph:

```python
# tools/graph_search.py

async def graph_entity_search(query: str, session_id: str, k: int = 10) -> List[Dict]:
    """
    Vector search on entities.

    Returns entities similar to query with their relationships.
    """
    embedding = await embeddings.aembed_query(query)

    result = await driver.run("""
        CALL db.index.vector.queryNodes('entity_vector', $k, $embedding)
        YIELD node AS entity, score
        WHERE $session_id IN entity.session_ids
        OPTIONAL MATCH (entity)-[r]-(related:Entity)
        WHERE $session_id IN related.session_ids
        RETURN entity.id, entity.type, entity.description, score,
               collect({id: related.id, type: type(r)}) AS connections
    """, {"embedding": embedding, "k": k, "session_id": session_id})

    return result


# tools/community_search.py

async def graph_community_search(query: str, session_id: str, k: int = 5) -> List[Dict]:
    """
    Search community summaries for high-level topics.

    Returns communities + their key entities.
    """
    embedding = await embeddings.aembed_query(query)

    result = await driver.run("""
        CALL db.index.vector.queryNodes('community_vector', $k, $embedding)
        YIELD node AS community, score
        WHERE community.session_id = $session_id
        MATCH (community)<-[:IN_COMMUNITY]-(e:Entity)
        RETURN community.title, community.summary, score,
               collect(e.id) AS entities
    """, {"embedding": embedding, "k": k, "session_id": session_id})

    return result


# tools/graph_traverse.py

async def graph_traverse(entity_id: str, session_id: str, depth: int = 2) -> Dict:
    """
    Multi-hop traversal from an entity.

    Returns the subgraph around this entity.
    """
    result = await driver.run("""
        MATCH path = (start:Entity {id: $entity_id})-[*1..$depth]-(end:Entity)
        WHERE $session_id IN start.session_ids
        AND $session_id IN end.session_ids
        RETURN start, relationships(path), end
    """, {"entity_id": entity_id, "depth": depth, "session_id": session_id})

    return result
```

### Synthesis Node Uses These Tools

```python
class SynthesisNode:
    """Synthesis with graph tools"""

    async def __call__(self, state: GlobalState) -> GlobalState:
        query = state["query"]
        session_id = state["session_id"]

        # 1. Search communities for high-level structure
        communities = await graph_community_search(query, session_id)

        # 2. Search entities for detailed content
        entities = await graph_entity_search(query, session_id)

        # 3. Build context from graph
        context = self._build_context(communities, entities)

        # 4. Generate synthesis
        synthesis = await self.llm.ainvoke(f"""
        Based on this research knowledge graph:

        Communities (high-level topics):
        {communities}

        Key entities and their relationships:
        {entities}

        Generate a comprehensive synthesis answering: {query}
        """)

        return {**state, "final_response": synthesis}
```

---

## IMPLEMENTATION STATUS

### Completed Files

| File | Status | Description |
|------|--------|-------------|
| `nodes/graph_builder/__init__.py` | ✅ | Module exports |
| `nodes/graph_builder/chunker.py` | ✅ | FindingChunker - pulls findings from Context Engine |
| `nodes/graph_builder/extractor.py` | ✅ | ConceptExtractor - LLMGraphTransformer for concepts |
| `nodes/graph_builder/consolidator.py` | ✅ | SchemaConsolidator - normalizes entity types |
| `nodes/graph_builder/builder.py` | ✅ | Neo4jGraphBuilder - builds graph in Neo4j |
| `nodes/graph_builder/embedder.py` | ✅ | EntityEmbedder - creates entity/community embeddings |
| `nodes/graph_builder/community_detector.py` | ✅ | CommunityDetector - Leiden algorithm |
| `nodes/graph_builder/summarizer.py` | ✅ | CommunitySummarizer - LLM summaries |
| `nodes/graph_builder/node.py` | ✅ | Main orchestrator node |

### New Dependencies Required

```toml
# Add to pyproject.toml
"graphdatascience>=1.0.0",
"langchain-experimental>=0.0.50",
```

### Integration Steps

1. **Add GRAPH_BUILDER phase** to `constants.py` ✅

2. **Update research graph routing** in `graph/graph.py`:
   ```python
   # Add node
   from .nodes.graph_builder import graph_builder_node
   graph.add_node("graph_builder", graph_builder_node)

   # Update routing (after lab architect, before synthesis)
   # Boss → Lab Architect → graph_builder → Synthesis → Critic
   ```

3. **Flash Explorers should STOP writing to Neo4j**
   - Remove Neo4j entity extraction from flash_explorer.py
   - Only write to Postgres/MinIO

4. **Add tools for Synthesis/Critic** (TODO):
   - `tools/graph_entity_search.py`
   - `tools/graph_community_search.py`
   - `tools/graph_traverse.py`

### Testing

```bash
# Run deep research with graph builder
uv run backend/manage.py research run "your query" --enable-graph-builder

# Check Neo4j for graph
# MATCH (d:Document {session_id: 'xxx'})-[*1..2]-(n) RETURN n LIMIT 50
```

