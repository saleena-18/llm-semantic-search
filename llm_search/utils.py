def print_result(result):
    print("Result:", result)

def keyword_search(query, client):
    return client.collections.get("DemoCollection").query.bm25(
        query=query,
        limit=5
    )