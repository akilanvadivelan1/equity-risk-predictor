# Deploying S&P 500 Risk Radar to AWS

This guide launches the web app on AWS so it is reachable at a public URL. The
app is a single Python web server (`app.py`) that reads the pre-extracted risk
text from your S3 bucket and runs the comparison live per request.

## What the app needs at runtime

- **Python 3.9 or newer** and the packages in `requirements.txt` (boto3 is the
  important one for reading S3).
- **Read access to the S3 bucket** `snp500-risk-radar-10k-data` in `us-east-1`.
- **Environment variables** (all optional, sensible defaults built in):
  - `PORT` the port to listen on. The app reads this and binds `0.0.0.0`.
    App Runner sets it to 8080 automatically.
  - `RISK_S3_BUCKET` defaults to `snp500-risk-radar-10k-data`.
  - `RISK_S3_PREFIX` defaults to `risk-factors`.
  - `AWS_REGION` defaults to `us-east-1`.

The app never needs AWS access keys baked in. On AWS it should use an
**IAM role** attached to the service, which boto3 picks up automatically.

## Recommended: AWS App Runner

App Runner builds and runs the web server from your GitHub repo, gives you
HTTPS and a public URL, and scales for you. Least operational work.

### Step 1. Confirm the start command and port

App Runner runs a start command and expects the app to listen on the port in
`PORT`. Our `app.py` already reads `PORT` and binds `0.0.0.0`, so the start
command is simply:

```
python3 app.py
```

### Step 2. Create the service

1. AWS Console, open **App Runner**, then **Create service**.
2. **Source**: choose **Source code repository**, connect your GitHub account,
   pick `akilanvadivelan1/equity-risk-predictor` and the branch you want to
   deploy (for example `main` after you merge, or `landing-page-mockup` to test).
3. **Deployment settings**: Automatic (redeploys on every push) or Manual.
4. **Build settings**, configure manually:
   - Runtime: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `python3 app.py`
   - Port: `8080`
5. **Service settings**:
   - Add environment variables if you want to override defaults. At minimum you
     can leave them unset (defaults are correct). To be explicit:
     - `RISK_S3_BUCKET = snp500-risk-radar-10k-data`
     - `AWS_REGION = us-east-1`
6. Create the service. App Runner builds and gives you a URL like
   `https://xxxx.us-east-1.awsapprunner.com`.

### Step 3. Give the service read access to S3 (IAM instance role)

App Runner uses an **instance role** for the running app's AWS calls.

1. In IAM, create a policy (JSON below), name it `RiskRadarS3Read`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::snp500-risk-radar-10k-data",
        "arn:aws:s3:::snp500-risk-radar-10k-data/*"
      ]
    }
  ]
}
```

2. Create an IAM role for App Runner:
   - Trusted entity: **AWS service**, use case **App Runner** (the
     `tasks.apprunner.amazonaws.com` principal, this is the **instance** role,
     not the build/access role).
   - Attach the `RiskRadarS3Read` policy.
3. In your App Runner service, **Configuration**, set the **Instance role** to
   this role. Deploy again if prompted.

That is it. The app can now read the bucket, and you have a public HTTPS URL.

## Alternative: Amazon Lightsail (a small fixed-price box)

Good if you prefer a simple always-on server you control, at a flat monthly cost.

1. Lightsail, **Create instance**, Linux, blueprint **OS Only, Amazon Linux 2023**
   (or Ubuntu). Pick the smallest plan (about 5 to 10 US dollars per month).
2. Once running, connect via the browser SSH button, then:

```bash
sudo dnf install -y python3 python3-pip git      # Amazon Linux 2023
# (Ubuntu: sudo apt update && sudo apt install -y python3 python3-pip git)

git clone https://github.com/akilanvadivelan1/equity-risk-predictor.git
cd equity-risk-predictor
pip3 install -r requirements.txt

export PORT=8080
export RISK_S3_BUCKET=snp500-risk-radar-10k-data
export AWS_REGION=us-east-1
python3 app.py
```

3. **Credentials**: attach an IAM role to the Lightsail instance if available,
   or run `aws configure` with an IAM user that has the `RiskRadarS3Read`
   policy above. Never use root keys.
4. **Open the port**: in the Lightsail instance **Networking** tab, add a
   firewall rule for the app port (for example TCP 8080), or put it behind port
   80 or 443.
5. **Keep it running** after you disconnect. Use a process manager:

```bash
# simple approach with nohup
nohup python3 app.py > risk-radar.log 2>&1 &

# or a systemd service (recommended for restarts on reboot)
```

Example systemd unit at `/etc/systemd/system/risk-radar.service`:

```ini
[Unit]
Description=S&P 500 Risk Radar
After=network.target

[Service]
WorkingDirectory=/home/ec2-user/equity-risk-predictor
Environment=PORT=8080
Environment=RISK_S3_BUCKET=snp500-risk-radar-10k-data
Environment=AWS_REGION=us-east-1
ExecStart=/usr/bin/python3 app.py
Restart=always
User=ec2-user

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now risk-radar
```

For a proper domain and HTTPS on Lightsail, put Nginx or Caddy in front as a
reverse proxy, or use a Lightsail load balancer with a certificate.

## Alternative: EC2

Same steps as Lightsail (it is a plain Linux box). Prefer attaching an **IAM
instance role** with the `RiskRadarS3Read` policy so no keys live on the box,
open the port in the security group, and run under systemd as above.

## Updating the site after code changes

- **App Runner with automatic deploys**: just push to the connected branch. It
  rebuilds and redeploys.
- **App Runner manual**: click **Deploy** in the console.
- **Lightsail or EC2**:

```bash
cd equity-risk-predictor
git pull
pip3 install -r requirements.txt   # only if dependencies changed
sudo systemctl restart risk-radar  # or restart your nohup process
```

## Refreshing the data

The web app reads whatever is in S3. To add new filings or a new year, run the
bulk downloader again from your machine (it is resumable and skips what is
already stored):

```bash
python3 download_all_to_s3.py
```

The site picks up the new files automatically on the next request. No redeploy
of the web app is needed for data changes.

## Cost and scaling notes

- The dataset is small (roughly 20 to 150 MB of text in S3), so S3 cost is
  pennies per month.
- App Runner bills for the running instance and requests. Lightsail is a flat
  monthly price. For a low-traffic learning project, either is inexpensive.
- The analysis runs live per request today. If traffic grows or pages feel
  slow, the natural next step is to precompute each company's analysis once and
  store the finished JSON in S3, then have the app just read it. The UI would
  not change.

## Quick checklist

- [ ] IAM policy `RiskRadarS3Read` created (GetObject, ListBucket on the bucket)
- [ ] Instance or service role attached with that policy
- [ ] Service started with `python3 app.py`, listening on `PORT`
- [ ] Env vars set or defaults confirmed (bucket, region)
- [ ] Public URL opens the landing page, and Analyze works for a ticker like AMZN
- [ ] Not investment advice disclaimer is visible (it is built into the pages)
