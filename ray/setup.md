# Setting up the Dashboard
You need Prometheus and Grafana to visualize time series data in the Ray dashboard.

# Prometheus
`brew install prometheus`
After installation, I can do:
`prometheus --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml`

Prometheus is displayed at http://localhost:9090 . I still need to figure out why Grafana integeration is needed. But this works for now. 


# Grafana
`brew install grafana`