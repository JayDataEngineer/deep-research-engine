"""Vector Manager CLI Plugin

Provides CLI commands for managing vector collections and RAG.

Usage:
    uv run backend/manage.py vectors list
    uv run backend/manage.py vectors collections
    uv run backend/manage.py vectors search <query>
    uv run backend/manage.py vectors add <collection> <text>
"""

import argparse
import logging

logger = logging.getLogger(__name__)


def cmd_list(args):
    """List all vector collections."""
    from src.shared.tools.registry import load_tools_by_category
    import asyncio

    try:
        tools = load_tools_by_category("memory")

        # Find list_collections tool
        list_tool = None
        for tool in tools:
            if hasattr(tool, 'name') and 'list_collection' in tool.name:
                list_tool = tool
                break
            elif hasattr(tool, '__name__') and 'list_collection' in tool.__name__:
                list_tool = tool
                break

        if list_tool:
            # Handle LangChain StructuredTool - use invoke or run the underlying async function
            if hasattr(list_tool, 'ainvoke'):
                result = asyncio.run(list_tool.ainvoke({}))
            elif hasattr(list_tool, 'invoke'):
                result = list_tool.invoke({})
            elif asyncio.iscoroutinefunction(list_tool):
                result = asyncio.run(list_tool())
            else:
                result = list_tool()

            print(f"\nVector Collections:")
            if isinstance(result, list) and result:
                for coll in result:
                    print(f"  - {coll}")
            elif isinstance(result, str):
                print(result)
            else:
                print(result)
        else:
            print("\nVector Collections:")
            print("  No collections found (vector tools may not be available)")

        return {"collections": result if list_tool else []}

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        print(f"Error: {e}")
        return {"error": str(e)}


def cmd_search(args):
    """Search vectors."""
    from src.shared.tools.registry import get_tool
    import asyncio

    try:
        search_tool = get_tool("vector_search")
        if not search_tool:
            print("Vector search tool not available")
            return {"error": "Tool not available"}

        print(f"\nSearching vectors for: {args.query}")

        # Handle LangChain StructuredTool
        if hasattr(search_tool, 'ainvoke'):
            result = asyncio.run(search_tool.ainvoke({"query": args.query, "top_k": 10}))
        elif hasattr(search_tool, 'invoke'):
            result = search_tool.invoke({"query": args.query, "top_k": 10})
        elif asyncio.iscoroutinefunction(search_tool):
            result = asyncio.run(search_tool(query=args.query, top_k=10))
        else:
            result = search_tool(query=args.query, top_k=10)

        print(f"\nResults:")
        if isinstance(result, list):
            for i, item in enumerate(result[:10], 1):  # Show top 10
                print(f"\n  [{i}] {item.get('content', '')[:100]}...")
                if 'metadata' in item:
                    print(f"      Metadata: {item['metadata']}")
        else:
            print(result)

        return {"results": result}

    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        print(f"Error: {e}")
        return {"error": str(e)}


def cmd_add(args):
    """Add documents to a collection."""
    from src.shared.tools.registry import get_tool
    import asyncio

    try:
        add_tool = get_tool("add_documents")
        if not add_tool:
            print("Add documents tool not available")
            return {"error": "Tool not available"}

        print(f"\nAdding document to collection: {args.collection}")

        # If text_file is provided, read from file
        if args.text_file:
            with open(args.text_file, 'r') as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            print("Error: Please provide --text or --text-file")
            return {"error": "No text provided"}

        # Handle LangChain StructuredTool
        if hasattr(add_tool, 'ainvoke'):
            result = asyncio.run(add_tool.ainvoke({"texts": [text]}))
        elif hasattr(add_tool, 'invoke'):
            result = add_tool.invoke({"texts": [text]})
        elif asyncio.iscoroutinefunction(add_tool):
            result = asyncio.run(add_tool(texts=[text]))
        else:
            result = add_tool(texts=[text])

        print(f"\nResult: {result}")
        return {"result": result}

    except FileNotFoundError:
        print(f"Error: File not found: {args.text_file}")
        return {"error": "File not found"}
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        print(f"Error: {e}")
        return {"error": str(e)}


def cmd_collections(args):
    """List all collections (alias for list)."""
    return cmd_list(args)


def register(parser):
    """Register CLI commands with the parser.

    Args:
        parser: An ArgumentParser that we add subparsers to
    """
    # Add subparsers to the parser
    subparsers = parser.add_subparsers(
        dest="vectors_subcommand",
        help="Vector management commands"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List vector collections")
    list_parser.set_defaults(func=cmd_list)

    # collections command (alias)
    col_parser = subparsers.add_parser("collections", help="List vector collections")
    col_parser.set_defaults(func=cmd_collections)

    # search command
    search_parser = subparsers.add_parser("search", help="Search vectors")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--collection", "-c", help="Collection name", default="default")
    search_parser.set_defaults(func=cmd_search)

    # add command
    add_parser = subparsers.add_parser("add", help="Add document to collection")
    add_parser.add_argument("collection", help="Collection name")
    add_parser.add_argument("--text", "-t", help="Text to add")
    add_parser.add_argument("--text-file", "-f", help="File containing text to add")
    add_parser.set_defaults(func=cmd_add)
