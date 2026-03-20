import asyncio
import time
from typing import Any, List, Dict

class MockCursor:
    def __init__(self, docs):
        self.docs = docs
        self.index = 0

    def limit(self, n):
        self.docs = self.docs[:n]
        return self

    def sort(self, field, order):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.docs):
            doc = self.docs[self.index]
            self.index += 1
            return doc
        raise StopAsyncIteration

    async def to_list(self, length):
        return self.docs[:length]

class MockCollection:
    def __init__(self, docs):
        self.docs = docs

    def find(self, filter_dict, projection=None):
        results = []
        for doc in self.docs:
            match = True
            for k, v in filter_dict.items():
                if k == "$or":
                    match = any(doc.get(sub_k) == sub_v for sub_doc in v for sub_k, sub_v in sub_doc.items())
                elif doc.get(k) != v:
                    match = False
                    break
            if match:
                res = doc.copy()
                if projection:
                    if projection.get("_id") == 0:
                        res.pop("_id", None)
                    # Handle inclusion projections
                    inclusion = {k: v for k, v in projection.items() if v == 1}
                    if inclusion:
                        new_res = {}
                        for k in inclusion:
                            if k in res:
                                new_res[k] = res[k]
                        res = new_res
                results.append(res)
        return MockCursor(results)

    async def find_one(self, filter_dict, projection=None):
        cursor = self.find(filter_dict, projection)
        results = await cursor.to_list(1)
        return results[0] if results else None

async def original_query_logic(collection, filter_dict, limit):
    # This simulates what the original query code did
    documents = await collection.find(filter_dict).limit(limit).to_list(length=limit)
    results = []
    for doc in documents:
        doc.pop("_id", None)
        results.append(doc)
    return results

async def optimized_query_logic(collection, filter_dict, limit):
    # This simulates what the optimized query code does
    documents = await collection.find(filter_dict, {"_id": 0}).limit(limit).to_list(length=limit)
    return documents

async def original_list_items_logic(collection, namespace, limit):
    cursor = (
        collection.find(
            {"namespace": namespace},
            {"id": 1, "namespace": 1, "type": 1, "created_at": 1, "metadata": 1},
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    items = []
    async for doc in cursor:
        doc.pop("_id", None)
        items.append(doc)
    return items

async def optimized_list_items_logic(collection, namespace, limit):
    cursor = (
        collection.find(
            {"namespace": namespace},
            {
                "_id": 0,
                "id": 1,
                "namespace": 1,
                "type": 1,
                "created_at": 1,
                "metadata": 1,
            },
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    return [doc async for doc in cursor]

async def benchmark():
    num_docs = 10000
    docs = [
        {"_id": i, "id": str(i), "namespace": "test", "type": "doc", "created_at": i, "metadata": {"key": "val"}}
        for i in range(num_docs)
    ]

    collection = MockCollection(docs)
    filter_dict = {"namespace": "test"}
    limit = num_docs

    print(f"Benchmarking with {num_docs} documents...")

    # Query benchmark
    start_time = time.perf_counter()
    res_orig_query = await original_query_logic(collection, filter_dict, limit)
    query_orig_duration = time.perf_counter() - start_time

    start_time = time.perf_counter()
    res_opt_query = await optimized_query_logic(collection, filter_dict, limit)
    query_opt_duration = time.perf_counter() - start_time

    print(f"Query - Original: {query_orig_duration:.6f}s, Optimized: {query_opt_duration:.6f}s")
    assert res_orig_query == res_opt_query

    # List items benchmark
    start_time = time.perf_counter()
    res_orig_list = await original_list_items_logic(collection, "test", limit)
    list_orig_duration = time.perf_counter() - start_time

    start_time = time.perf_counter()
    res_opt_list = await optimized_list_items_logic(collection, "test", limit)
    list_opt_duration = time.perf_counter() - start_time

    print(f"List Items - Original: {list_orig_duration:.6f}s, Optimized: {list_opt_duration:.6f}s")
    assert res_orig_list == res_opt_list

    print("Success: Results are identical.")

if __name__ == "__main__":
    asyncio.run(benchmark())
