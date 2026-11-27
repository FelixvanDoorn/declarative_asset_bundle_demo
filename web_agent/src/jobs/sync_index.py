"""Create the Delta Sync vector index on wiki_chunks, or sync it if it already exists.

Runs after the pipeline in the wiki_index job, because a Delta Sync index can only be
created once its source table exists. Embeddings are self-managed (computed with ai_query
in the pipeline), so the index points at the `embedding` column.
"""

import argparse
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    PipelineType,
    VectorIndexType,
)

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", required=True)
parser.add_argument("--index", required=True)
parser.add_argument("--source-table", required=True)
parser.add_argument("--embedding-dimension", type=int, default=1024)
args = parser.parse_args()

w = WorkspaceClient()

print(f"Waiting for vector search endpoint {args.endpoint} to be online...")
w.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(args.endpoint)

try:
    w.vector_search_indexes.get_index(args.index)
    print(f"Index {args.index} exists; triggering a sync.")
    w.vector_search_indexes.sync_index(args.index)
except NotFound:
    print(f"Creating index {args.index} on {args.source_table}.")
    w.vector_search_indexes.create_index(
        name=args.index,
        endpoint_name=args.endpoint,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=args.source_table,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_vector_columns=[
                EmbeddingVectorColumn(name="embedding", embedding_dimension=args.embedding_dimension)
            ],
            columns_to_sync=["chunk_id", "article_id", "title", "chunk_index", "chunk", "embedding"],
        ),
    )

# Wait (up to 30 min) until the index is ready to serve queries.
deadline = time.time() + 30 * 60
while time.time() < deadline:
    status = w.vector_search_indexes.get_index(args.index).status
    print(f"ready={status.ready} rows={status.indexed_row_count} {status.message or ''}")
    if status.ready:
        break
    time.sleep(30)
else:
    raise TimeoutError(f"Index {args.index} not ready after 30 minutes")
