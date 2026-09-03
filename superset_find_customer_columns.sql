SELECT table_schema, table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'test'
  AND (
    LOWER(column_name) LIKE '%customer%'
    OR LOWER(column_name) LIKE '%client%'
    OR LOWER(column_name) LIKE '%business%'
  )
ORDER BY table_name, column_name
LIMIT 500
