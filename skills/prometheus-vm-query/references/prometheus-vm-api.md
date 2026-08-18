# VictoriaMetrics / Prometheus API Reference

## Default Endpoint

Use this base URL unless the user provides another tenant or endpoint:

```text
https://vm-select.mvm.qiniu.io/select/293:0/prometheus/api/v1
```

The helper script posts form-encoded requests to avoid URL length limits:

- Instant query: `POST /query`
- Range query: `POST /query_range`
- Label names: `POST /labels`
- Label values: `POST /label/<label>/values`
- Series lookup: `POST /series`

## Time Handling

Prometheus APIs evaluate times in UTC. Grafana/VMUI may display Asia/Shanghai.

Use script options for Beijing wall-clock input:

```bash
--beijing-time "2026-06-04 20:00:00"
--beijing-start "2026-06-04 19:00:00"
--beijing-end "2026-06-04 20:00:00"
```

The script converts these to UTC RFC3339 before sending them to VM.

## Script Operations

Instant:

```bash
scripts/vm_promql_query.py instant --query '<promql>' --beijing-time '<YYYY-MM-DD HH:MM:SS>'
```

Range:

```bash
scripts/vm_promql_query.py range --query '<promql>' --beijing-start '<time>' --beijing-end '<time>' --step 1m
```

Label values:

```bash
scripts/vm_promql_query.py label-values node_id --match 'jarvis_node_bw_current:join{customer_id="1380317970"}'
```

Series lookup:

```bash
scripts/vm_promql_query.py series --match 'jarvis_node_bw_current:join{node_id="..."}'
```

## Output

Use `--output out.csv` or `--output out.json`. The script infers output format from the suffix.

Flattening rules:

- Vector result: one row per series.
- Matrix result: one row per sample per series.
- Labels become columns.
- Every numeric sample includes `timestamp`, `time_utc`, `time_bj`, and `value`.

Preview rows are printed to stdout unless `--preview 0` is used.

## Batch Node Query

For many nodes, put IDs in a text or CSV file and use a placeholder in PromQL:

```promql
node_id=~"__NODE_REGEX__"
```

Command:

```bash
scripts/vm_promql_query.py instant \
  --node-file nodes.txt \
  --node-placeholder __NODE_REGEX__ \
  --node-chunk-size 80 \
  --query 'sum by (node_id) (jarvis_node_bw_current:join{node_id=~"__NODE_REGEX__"})'
```

The script replaces the placeholder with a `|`-joined regex for each chunk and merges the rows.

## Guardrails

Avoid unrestricted queries on high-cardinality metrics, especially:

```promql
niulink_agent_flow_upstream
```

Add one of these filters before querying:

- `node_id`
- `switch_id`
- `idc_id`
- `province`
- `isp`

Use `--allow-wide-query` only when intentionally running a broad aggregate and accepting latency/series cost.
