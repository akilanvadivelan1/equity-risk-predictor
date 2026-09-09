#!/usr/bin/env python3
"""
S&P 500 Risk Radar - Precompute Analysis to S3
==============================================
Computes each company's full analysis once (risk language + financial health +
combined verdict) and stores it as analysis/<TICKER>.json in S3. The web app
reads that file directly, so page loads become near-instant instead of running
the full pipeline on every request.

Run this after the bulk filing downloader, and again whenever you refresh the
data or change the scoring logic.

Usage:
    pip install boto3
    aws sso login --profile NAME   (or aws configure)   # active AWS creds
    python3 precompute_to_s3.py

    If it stops, just run it again. It resumes (skips tickers already stored,
    unless you pass --force).

Options:
    --force            Recompute and overwrite even if analysis/<TICKER>.json exists
    --tickers A,B,C    Only these tickers (comma separated)
    --limit N          Only the first N companies not yet done

AWS credentials use standard boto3 resolution. The IAM identity needs
s3:GetObject, s3:PutObject, and s3:ListBucket on the bucket.
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the exact analysis builder and S3 helpers the web app uses, so the
# precomputed output is identical to a live computation.
import app


def _existing_analysis_keys(s3, bucket):
    """Return the set of tickers that already have analysis/<TICKER>.json."""
    done = set()
    token = None
    while True:
        kwargs = {'Bucket': bucket, 'Prefix': 'analysis/'}
        if token:
            kwargs['ContinuationToken'] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):
                done.add(key[len('analysis/'):-len('.json')].upper())
        if resp.get('IsTruncated'):
            token = resp.get('NextContinuationToken')
        else:
            break
    return done


def main():
    parser = argparse.ArgumentParser(description="Precompute Risk Radar analyses to S3.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing analyses")
    parser.add_argument('--tickers', type=str, default=None, help="Comma-separated tickers only")
    parser.add_argument('--limit', type=int, default=None, help="Only the first N not-yet-done")
    args = parser.parse_args()

    s3 = app._get_s3_client()
    if s3 is None:
        print("ERROR: boto3/S3 not available. Install boto3 and configure AWS credentials.")
        sys.exit(1)

    bucket = app.RISK_S3_BUCKET
    print("=" * 60)
    print("S&P 500 Risk Radar - precompute analyses to S3")
    print(f"Bucket: {bucket}")
    print("=" * 60)

    # Which companies to process (those with >=2 stored years).
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    else:
        companies = app.list_companies()
        tickers = [c['ticker'] for c in companies]
    print(f"Candidate companies: {len(tickers)}")

    done = set() if args.force else _existing_analysis_keys(s3, bucket)
    if done:
        print(f"Already precomputed: {len(done)} (skipping; use --force to redo).")

    todo = [t for t in tickers if args.force or t not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"To compute this run: {len(todo)}")
    print("-" * 60)

    stored = failed = 0
    for i, ticker in enumerate(todo, start=1):
        try:
            result = app.build_analysis(ticker)
            if result.get('error'):
                print(f"[{i}/{len(todo)}] {ticker}: skipped ({result['error']})")
                failed += 1
                continue
            if app.write_precomputed_analysis(ticker, result):
                changed = len(result.get('risks', []))
                fin = 'fin' if result.get('fundamentals', {}).get('available') else 'no-fin'
                print(f"[{i}/{len(todo)}] {ticker}: stored ({changed} changed risks, {fin})")
                stored += 1
            else:
                print(f"[{i}/{len(todo)}] {ticker}: FAILED to write to S3")
                failed += 1
        except Exception as e:
            print(f"[{i}/{len(todo)}] {ticker}: ERROR {e}")
            failed += 1
        # Gentle pause. build_analysis makes a few SEC calls for fundamentals.
        time.sleep(0.5)

    print("-" * 60)
    print(f"DONE. Stored: {stored}   Failed/skipped: {failed}")
    print("Re-run anytime to fill in the rest. The web app now serves these instantly.")


if __name__ == "__main__":
    main()
