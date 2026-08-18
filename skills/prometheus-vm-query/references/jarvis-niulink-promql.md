# Jarvis / NiuLink PromQL Recipes

## Node Business Bandwidth

`jarvis_node_bw_current:join` is already in Mbps.

Node/business bandwidth:

```promql
sum by (node_id, node, customer_id, customer_zh, province, isp, vendor_id) (
  jarvis_node_bw_current:join{
    node_type="node",
    delivery_type!="idc",
    stage="inService"
  }
)
```

Kuaishou VOD business example:

```promql
sum by (node_id, customer_id, customer_zh) (
  jarvis_node_bw_current:join{customer_id="1380317970"}
)
```

## Upstream Access Province / ISP Flow

`niulink_agent_flow_upstream` currently behaves as a byte counter.

Convert to Mbps:

```promql
clamp_min(rate(niulink_agent_flow_upstream{node_id="<node_id>",province!=""}[5m]), 0) * 8 / 1000 / 1000
```

Province traffic Top20:

```promql
topk(20,
  sum by (province) (
    clamp_min(rate(niulink_agent_flow_upstream{node_id="$node",province!=""}[5m]), 0)
  ) * 8 / 1000 / 1000
)
```

Province+ISP traffic:

```promql
topk(50,
  sum by (province, isp) (
    clamp_min(rate(niulink_agent_flow_upstream{node_id="$node",province!="",isp!=""}[5m]), 0)
  ) * 8 / 1000 / 1000
)
```

Province share percentage:

```promql
topk(20,
  (
    sum by (province) (
      clamp_min(rate(niulink_agent_flow_upstream{node_id="$node",province!=""}[5m]), 0)
    )
    /
    clamp_min(
      sum(clamp_min(rate(niulink_agent_flow_upstream{node_id="$node",province!=""}[5m]), 0)),
      1
    )
  ) * 100
)
```

Batch node placeholder:

```promql
sum by (node_id, province, isp) (
  clamp_min(rate(niulink_agent_flow_upstream{node_id=~"__NODE_REGEX__",province!="",isp!=""}[5m]), 0)
) * 8 / 1000 / 1000
```

## Retransmission

Node-level retransmission percentage:

```promql
jarvis_node_tcp_retran_ratio_5m{node_id="<node_id>"} * 100
```

Important limitation: this metric is node-level and does not contain `province` or `isp`.

For "which service province caused retransmission to rise", use a correlation view:

```promql
topk(20,
  (
    sum by (province, isp) (
      clamp_min(rate(niulink_agent_flow_upstream{node_id="$node",province!="",isp!=""}[5m]), 0)
    ) * 8 / 1000 / 1000
  )
  * on() group_left()
  (jarvis_node_tcp_retran_ratio_5m{node_id="$node"} * 100)
)
```

This is a suspect score, not proof of province-level retransmission. A direct answer needs a metric that carries both retransmission and service province/ISP labels.

## Drop Alert Pattern

Large non-IDC in-service node, historical baseline high, current 25-minute window below 20% of 5-hour baseline:

```promql
(
  max_over_time((
    avg_over_time((
      sum by (node_id, node, customer_id, customer_zh, province, isp, vendor_id) (
        jarvis_node_bw_current:join{
          node_type="node",
          delivery_type!="idc",
          stage="inService"
        }
      )
    )[5m:1m])
  )[5h:1m]) > 250
)
and
(
  (
    max_over_time((
      sum by (node_id, node, customer_id, customer_zh, province, isp, vendor_id) (
        jarvis_node_bw_current:join{
          node_type="node",
          delivery_type!="idc",
          stage="inService"
        }
      )
    )[25m:1m])
    or
    max_over_time((
      avg_over_time((
        sum by (node_id, node, customer_id, customer_zh, province, isp, vendor_id) (
          jarvis_node_bw_current:join{
            node_type="node",
            delivery_type!="idc",
            stage="inService"
          }
        )
      )[5m:1m])
    )[5h:1m]) * 0
  )
  <
  max_over_time((
    avg_over_time((
      sum by (node_id, node, customer_id, customer_zh, province, isp, vendor_id) (
        jarvis_node_bw_current:join{
          node_type="node",
          delivery_type!="idc",
          stage="inService"
        }
      )
    )[5m:1m])
  )[5h:1m]) * 0.2
)
```

## Business Scope Notes

- `jarvis_node_bw_current:join` has business labels such as `customer_id` and `customer_zh`.
- `niulink_agent_flow_upstream` has service access labels such as `province` and `isp`, but no business label.
- When a business-scoped upstream report is required, first derive the node list from Superset or `jarvis_node_bw_current:join`, then query `niulink_agent_flow_upstream` by those `node_id` values.
