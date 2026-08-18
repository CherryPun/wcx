---
name: prometheus-vm-query
description: General VictoriaMetrics/Prometheus query skill for running instant/range PromQL, label discovery, safe high-cardinality queries, Beijing-time evaluation, node_id batch chunking, and CSV/JSON exports. Use when Codex needs to query VM/Prometheus metrics, validate Jarvis/NiuLink metric units, fetch node bandwidth/retransmission/upstream province-ISP traffic, or turn PromQL into reusable operational data.
---

# Prometheus VM Query

## Governance
- Layer: `primitive`
- Level: `L2`
- Domain: `monitoring`, `victoriametrics`, `prometheus`, `jarvis`, `niulink`
- Risk: `read`
- Owner: `pending-assignment`
- Status: `active`
- Last Verified: `2026-06-05`

## Overview
Use this skill to query VictoriaMetrics/Prometheus safely and turn PromQL results into inspectable tables, CSV, or JSON.

Default VM API base:

```text
https://vm-select.mvm.qiniu.io/select/293:0/prometheus/api/v1
```

## Workflow
1. Clarify the query shape:
   - Instant value at a fixed time, such as Beijing 20:00.
   - Range trend over a window.
   - Label discovery or metric series inspection.
   - Batch query by `node_id`.
2. Prefer `scripts/vm_promql_query.py` for live calls and exports.
3. Convert Beijing wall-clock times before calling the API:
   - Use `--beijing-time`, `--beijing-start`, and `--beijing-end`.
   - Beijing `2026-06-04 20:00:00` becomes UTC `2026-06-04T12:00:00Z`.
4. Protect high-cardinality metrics:
   - Do not query `niulink_agent_flow_upstream` without a narrow label filter.
   - Filter by `node_id`, `switch_id`, `idc_id`, `province`, or `isp`, or use batch chunking.
   - Use `--allow-wide-query` only after intentionally accepting the cost.
5. For Jarvis/NiuLink metric semantics and ready PromQL, read `references/jarvis-niulink-promql.md`.
6. For API details, command examples, and output rules, read `references/prometheus-vm-api.md`.

## Quick Commands
Instant query at Beijing 20:00:

```bash
/Users/rj/Desktop/skills/shared/primitives/prometheus-vm-query/scripts/vm_promql_query.py instant \
  --beijing-time "2026-06-04 20:00:00" \
  --query 'sum(jarvis_node_bw_current:join{node_id="03cf3818dcef1f6b24b1605c518ded80"})'
```

Range query for a trend:

```bash
/Users/rj/Desktop/skills/shared/primitives/prometheus-vm-query/scripts/vm_promql_query.py range \
  --beijing-start "2026-06-04 19:00:00" \
  --beijing-end "2026-06-04 20:00:00" \
  --step 1m \
  --query 'sum(jarvis_node_bw_current:join{node_id="03cf3818dcef1f6b24b1605c518ded80"})'
```

Batch nodes with a regex placeholder:

```bash
/Users/rj/Desktop/skills/shared/primitives/prometheus-vm-query/scripts/vm_promql_query.py instant \
  --beijing-time "2026-06-04 20:00:00" \
  --node-file nodes.txt \
  --node-placeholder __NODE_REGEX__ \
  --node-chunk-size 80 \
  --output out.csv \
  --query 'sum by (node_id,province,isp) (clamp_min(rate(niulink_agent_flow_upstream{node_id=~"__NODE_REGEX__",province!="",isp!=""}[5m]),0)) * 8 / 1000 / 1000'
```

Discover label values:

```bash
/Users/rj/Desktop/skills/shared/primitives/prometheus-vm-query/scripts/vm_promql_query.py label-values customer_id \
  --match 'jarvis_node_bw_current:join{customer_id!=""}'
```

## Common Metric Rules
- `jarvis_node_bw_current:join` is already Mbps.
- `niulink_agent_flow_upstream` is a byte counter in current evidence. Use `clamp_min(rate(...[5m]), 0) * 8 / 1000 / 1000` for Mbps.
- `jarvis_node_tcp_retran_ratio_5m * 100` gives node-level retransmission percentage; it does not directly identify service province.
- `niulink_agent_flow_upstream` has province/ISP labels but no business labels such as `customer_id`.

## Output Expectations
When returning query results to a user:
- Include the evaluation time and timezone.
- State units explicitly.
- If the query is batched or truncated, mention the chunking or preview limit.
- For operational reports, export both detail and summary CSV/JSON when useful.

## References
- `references/prometheus-vm-api.md`: API usage, CLI options, safe querying, time conversion.
- `references/jarvis-niulink-promql.md`: common Jarvis/NiuLink metrics and PromQL recipes.
