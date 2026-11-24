-- Fixed-size, overlapping character chunks with an embedding per chunk (computed once per
-- chunk with ai_query, since this is a streaming table). This table is the source of the
-- Delta Sync vector index, which requires Change Data Feed.
CREATE OR REFRESH STREAMING TABLE wiki_chunks
COMMENT 'Chunked Wikipedia articles with embeddings for vector search'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
AS
WITH chunked AS (
  SELECT
    article_id,
    title,
    chunk_index,
    substring(text, chunk_index * (${wiki.chunk_size} - ${wiki.chunk_overlap}) + 1, ${wiki.chunk_size}) AS chunk
  FROM (
    SELECT
      article_id,
      title,
      text,
      explode(sequence(0, greatest(
        CAST(ceil((length(text) - ${wiki.chunk_overlap}) / (${wiki.chunk_size} - ${wiki.chunk_overlap})) AS INT) - 1,
        0
      ))) AS chunk_index
    FROM STREAM(wiki_articles)
  )
)
SELECT
  concat(article_id, '-', chunk_index) AS chunk_id,
  article_id,
  title,
  chunk_index,
  chunk,
  -- Embed the title together with the chunk so short chunks keep their context.
  ai_query('${wiki.embedding_endpoint}', concat(title, ': ', chunk)) AS embedding
FROM chunked;
