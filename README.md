# QUERIES

## p mod 5 all models combined

```
data = from(bucket: "metrics")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "request_status_time")
  |> filter(fn: (r) => r["_field"] == "request_status_time")
  |> filter(fn: (r) => r["status"] == "RUNNING")
  |> group()

p5 = data |> quantile(q: 0.05) |> set(key: "percentile", value: "p5")
p10 = data |> quantile(q: 0.10) |> set(key: "percentile", value: "p10")
p15 = data |> quantile(q: 0.15) |> set(key: "percentile", value: "p15")
p20 = data |> quantile(q: 0.20) |> set(key: "percentile", value: "p20")
p25 = data |> quantile(q: 0.25) |> set(key: "percentile", value: "p25")
p30 = data |> quantile(q: 0.30) |> set(key: "percentile", value: "p30")
p35 = data |> quantile(q: 0.35) |> set(key: "percentile", value: "p35")
p40 = data |> quantile(q: 0.40) |> set(key: "percentile", value: "p40")
p45 = data |> quantile(q: 0.45) |> set(key: "percentile", value: "p45")
p50 = data |> quantile(q: 0.50) |> set(key: "percentile", value: "p50")
p55 = data |> quantile(q: 0.55) |> set(key: "percentile", value: "p55")
p60 = data |> quantile(q: 0.60) |> set(key: "percentile", value: "p60")
p65 = data |> quantile(q: 0.65) |> set(key: "percentile", value: "p65")
p70 = data |> quantile(q: 0.70) |> set(key: "percentile", value: "p70")
p75 = data |> quantile(q: 0.75) |> set(key: "percentile", value: "p75")
p80 = data |> quantile(q: 0.80) |> set(key: "percentile", value: "p80")
p85 = data |> quantile(q: 0.85) |> set(key: "percentile", value: "p85")
p90 = data |> quantile(q: 0.90) |> set(key: "percentile", value: "p90")
p95 = data |> quantile(q: 0.95) |> set(key: "percentile", value: "p95")
p100 = data |> quantile(q: 1.00) |> set(key: "percentile", value: "p100")

union(tables: [p5, p10, p15, p20, p25, p30, p35, p40, p45, p50, p55, p60, p65, p70, p75, p80, p85, p90, p95, p100])
  |> group()
  |> keep(columns: ["percentile", "_value"])
  |> rename(columns: {_value: "runtime_seconds"})
```

## Percentiles by model p50 p90 p99

```
data = from(bucket: "metrics")
  |> range(start: -30d)
  |> filter(fn: (r) => r["_measurement"] == "request_status_time")
  |> filter(fn: (r) => r["_field"] == "request_status_time")
  |> filter(fn: (r) => r["status"] == "RUNNING")
  |> group(columns: ["model_key"])

p50 = data |> quantile(q: 0.50) |> set(key: "percentile", value: "p50")
p90 = data |> quantile(q: 0.90) |> set(key: "percentile", value: "p90")
p99 = data |> quantile(q: 0.99) |> set(key: "percentile", value: "p99")

union(tables: [p50, p90, p99])
  |> group()
  |> keep(columns: ["model_key", "percentile", "_value"])
  |> rename(columns: {_value: "runtime_seconds"})
  ```