-- Raw Wikipedia articles from one file of the samples dataset.
-- The files are gzipped TSV: id, title, revision timestamp, text, contributor. They are read
-- as plain text and split on tabs, because article text contains quotes that break CSV parsing.
-- Only articles with id % sample_modulo = 0 are kept, to limit embedding cost (1 = all).
CREATE OR REFRESH STREAMING TABLE wiki_articles
COMMENT 'Wikipedia articles from one samples file (sampled by id)'
AS
SELECT
  CAST(cols[0] AS BIGINT)    AS article_id,
  cols[1]                    AS title,
  CAST(cols[2] AS TIMESTAMP) AS revision_ts,
  trim(cols[3])              AS text
FROM (
  SELECT split(value, '\t') AS cols
  FROM STREAM read_files(
    '/Volumes/samples/databricks/datasets/wiki/',
    format => 'text',
    pathGlobFilter => '${wiki.source_file}'
  )
)
WHERE size(cols) >= 4
  AND length(trim(cols[3])) > 0
  AND CAST(cols[0] AS BIGINT) % ${wiki.sample_modulo} = 0;
