# Setting up the Dashboard
You need Prometheus and Grafana to visualize time series data in the Ray dashboard.

# Prometheus
`brew install prometheus`
After installation, I need to point to Ray's prometheus file. On Mac, this step is a little different. You need to go into the default prometheus config file - for me this was /opt/homebrew/etc/prometheus.args and then add Ray's config file path `-config.file /tmp/ray/session_latest/metrics/prometheus/prometheus.yml` .


Prometheus is displayed at http://localhost:9090 . I still need to figure out why Grafana integeration is needed. But this works for now. 

I'm not too sure how folks verify their installations, but this is how "targets" looks like for me (i have a Ray instance running):
![Ray Prometheus](prometheus_ray.png)


# Grafana
`brew install grafana`

Once this is done, let's configure Grafana's config to work with Ray. The default config file path for me is `/opt/homebrew/etc/grafana/grafana.ini`. You need to copy the contents in Ray's Grafana config to this path:
`cp /tmp/ray/session_latest/metrics/grafana/grafana.ini /opt/homebrew/etc/grafana/grafana.ini`

Start grafana locally with 
`brew services start grafana`. 
Now, grafana should be running on `localhost:3000`
The default user name and password is `admin/admin`. You will be prompted to change your password after the first time.
Once you log in, make sure to update the profile details to your actual Grafana account.

With this, you should finally be able to see Ray Metrics!