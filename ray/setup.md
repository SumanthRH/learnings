# Setting up the Dashboard
You need Prometheus and Grafana to visualize time series data in the Ray dashboard.

# Prometheus
`brew install prometheus`
After installation, I can do:
`prometheus --config.file=/tmp/ray/session_latest/metrics/prometheus/prometheus.yml`

Prometheus is displayed at http://localhost:9090 . I still need to figure out why Grafana integeration is needed. But this works for now. 


# Grafana
`brew install grafana`
Start grafana locally with 
`brew services start grafana`. 
Now, grafana should be running on `localhost:3000`
The default user name and password is `admin/admin`. You will be prompted to change your password after the first time.
Once you log in, make sure to update the profile details to your actual Grafana account.
