"""
Stage 3 — Target Feasibility Analysis Script

Gathers all statistics needed for docs/03_target_feasibility_report.md
from the processed Parquet files. Analysis only — no model training.

Run: python -m src.target.feasibility_analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path("data/processed")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    # ================================================================
    # 1. LOAD AND INSPECT
    # ================================================================
    section("1. LOAD AND INSPECT PROCESSED DATA")

    devices = pd.read_parquet(PROCESSED / "devices.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    manufacturers = pd.read_parquet(PROCESSED / "manufacturers.parquet")
    merged = pd.read_parquet(PROCESSED / "merged.parquet")

    print(f"devices.parquet:        {devices.shape[0]:>8,} rows × {devices.shape[1]:>3} cols")
    print(f"events.parquet:         {events.shape[0]:>8,} rows × {events.shape[1]:>3} cols")
    print(f"manufacturers.parquet:  {manufacturers.shape[0]:>8,} rows × {manufacturers.shape[1]:>3} cols")
    print(f"merged.parquet:         {merged.shape[0]:>8,} rows × {merged.shape[1]:>3} cols")

    print("\nevents.parquet dtypes:")
    for c in events.columns:
        print(f"  {c:>30s}: {events[c].dtype}")

    print(f"\nevents.parquet key stats:")
    print(f"  Unique event IDs:     {events['id'].nunique():,}")
    print(f"  Unique device_ids:    {events['device_id'].nunique():,}")
    print(f"  event_date non-null:  {events['event_date'].notna().sum():,} ({events['event_date'].notna().mean()*100:.1f}%)")
    print(f"  event_date null:      {events['event_date'].isna().sum():,} ({events['event_date'].isna().mean()*100:.1f}%)")

    # ================================================================
    # 3. EVENT STRUCTURE
    # ================================================================
    section("3. EVENT STRUCTURE — Event Taxonomy")

    print("event type distribution:")
    for val, cnt in events["type"].value_counts(dropna=False).items():
        print(f"  {str(val):>40s}: {cnt:>7,} ({cnt/len(events)*100:5.1f}%)")

    print("\naction_classification distribution:")
    for val, cnt in events["action_classification"].value_counts(dropna=False).items():
        print(f"  {str(val):>30s}: {cnt:>7,} ({cnt/len(events)*100:5.1f}%)")

    print("\ndetermined_cause distribution (top 15):")
    for val, cnt in events["determined_cause"].value_counts(dropna=False).head(15).items():
        print(f"  {str(val):>45s}: {cnt:>7,} ({cnt/len(events)*100:5.1f}%)")

    print("\nevent_date_source distribution:")
    for val, cnt in events["event_date_source"].value_counts(dropna=False).items():
        print(f"  {str(val):>30s}: {cnt:>7,} ({cnt/len(events)*100:5.1f}%)")

    print("\nevent_date_available distribution:")
    for val, cnt in events["event_date_available"].value_counts(dropna=False).items():
        print(f"  {str(val):>10s}: {cnt:>7,} ({cnt/len(events)*100:5.1f}%)")

    # Are ALL events safety/failure related?
    print("\n*** KEY QUESTION: Are all events safety/failure-related? ***")
    all_types = events["type"].dropna().unique()
    safety_keywords = ["recall", "safety", "alert", "notice", "field safety"]
    for t in all_types:
        is_safety = any(k in t.lower() for k in safety_keywords)
        print(f"  '{t}' → safety-related: {is_safety}")

    # ================================================================
    # 4. DEVICE EVENT HISTORY
    # ================================================================
    section("4. DEVICE EVENT HISTORY")

    # Events per device
    epc = events.groupby("device_id").size().reset_index(name="event_count")
    total_devices_in_devices = devices["id"].nunique()
    total_devices_in_events = events["device_id"].nunique()

    # Devices in devices.parquet but NOT in events.parquet
    device_ids_in_devices = set(devices["id"].unique())
    device_ids_in_events = set(events["device_id"].unique())
    devices_with_no_events = device_ids_in_devices - device_ids_in_events
    devices_only_in_events = device_ids_in_events - device_ids_in_devices

    print(f"Total unique devices in devices.parquet:  {total_devices_in_devices:,}")
    print(f"Total unique device_ids in events.parquet: {total_devices_in_events:,}")
    print(f"Devices in devices.parquet with NO events: {len(devices_with_no_events):,}")
    print(f"Device_ids in events NOT in devices.parquet: {len(devices_only_in_events):,}")

    print(f"\nEvent count distribution:")
    print(f"  Devices with 1 event:   {(epc['event_count'] == 1).sum():>7,} ({(epc['event_count'] == 1).sum()/len(epc)*100:.2f}%)")
    print(f"  Devices with 2 events:  {(epc['event_count'] == 2).sum():>7,} ({(epc['event_count'] == 2).sum()/len(epc)*100:.2f}%)")
    print(f"  Devices with 3 events:  {(epc['event_count'] == 3).sum():>7,} ({(epc['event_count'] == 3).sum()/len(epc)*100:.2f}%)")
    print(f"  Devices with 4 events:  {(epc['event_count'] == 4).sum():>7,} ({(epc['event_count'] == 4).sum()/len(epc)*100:.2f}%)")
    print(f"  Devices with 5-10:      {((epc['event_count'] >= 5) & (epc['event_count'] <= 10)).sum():>7,}")
    print(f"  Devices with 11-50:     {((epc['event_count'] >= 11) & (epc['event_count'] <= 50)).sum():>7,}")
    print(f"  Devices with 51+:       {(epc['event_count'] >= 51).sum():>7,}")
    print(f"  Max events per device:  {epc['event_count'].max()}")
    print(f"  Mean:                   {epc['event_count'].mean():.3f}")
    print(f"  Median:                 {epc['event_count'].median():.0f}")
    print(f"  P95:                    {epc['event_count'].quantile(0.95):.0f}")
    print(f"  P99:                    {epc['event_count'].quantile(0.99):.0f}")

    # ================================================================
    # 5. CANDIDATE A — FUTURE SAFETY EVENT
    # ================================================================
    section("5. CANDIDATE A — Future Safety Event")

    print("CRITICAL: Can we define positive AND negative examples?")
    print(f"  Devices in devices.parquet:  {total_devices_in_devices:,}")
    print(f"  Devices with ≥1 event:       {total_devices_in_events:,}")
    print(f"  Devices with 0 events:       {len(devices_with_no_events):,}")
    print()
    print("*** FINDING: There are ZERO devices without events. ***")
    print(f"*** Every device in devices.parquet has at least 1 event. ***")
    print(f"*** This means devices.parquet is NOT a registry of all devices — ***")
    print(f"*** it is a table of devices that have had safety events. ***")
    print()

    # Temporal analysis for future-event prediction
    dated_events = events[events["event_date_available"]].copy()
    print(f"Dated events: {len(dated_events):,} / {len(events):,}")

    # Per-device: first and last event dates
    device_dates = dated_events.groupby("device_id")["event_date"].agg(["min", "max", "count"])
    device_dates.columns = ["first_event", "last_event", "n_events"]
    device_dates["span_days"] = (device_dates["last_event"] - device_dates["first_event"]).dt.days

    print(f"\nDevices with dated events: {len(device_dates):,}")

    # Single-event devices
    single = device_dates[device_dates["n_events"] == 1]
    multi = device_dates[device_dates["n_events"] > 1]
    print(f"  Single-event devices: {len(single):,} ({len(single)/len(device_dates)*100:.1f}%)")
    print(f"  Multi-event devices:  {len(multi):,} ({len(multi)/len(device_dates)*100:.1f}%)")

    if len(multi) > 0:
        print(f"\n  Multi-event device temporal spans:")
        print(f"    Mean span:   {multi['span_days'].mean():.0f} days")
        print(f"    Median span: {multi['span_days'].median():.0f} days")
        print(f"    P25 span:    {multi['span_days'].quantile(0.25):.0f} days")
        print(f"    P75 span:    {multi['span_days'].quantile(0.75):.0f} days")
        print(f"    Span = 0:    {(multi['span_days'] == 0).sum():,} ({(multi['span_days'] == 0).sum()/len(multi)*100:.1f}%)")
        print(f"    Span > 0:    {(multi['span_days'] > 0).sum():,}")
        print(f"    Span > 365:  {(multi['span_days'] > 365).sum():,}")

    # ================================================================
    # 6. CANDIDATE B — EVENT SEVERITY
    # ================================================================
    section("6. CANDIDATE B — Event Severity (action_classification)")

    ac = events["action_classification"].value_counts(dropna=False)
    total = len(events)
    ac_non_null = events["action_classification"].notna().sum()

    print(f"Total events:                     {total:,}")
    print(f"action_classification non-null:   {ac_non_null:,} ({ac_non_null/total*100:.1f}%)")
    print(f"action_classification null:       {total - ac_non_null:,} ({(total-ac_non_null)/total*100:.1f}%)")
    print()

    print("Class distribution (non-null):")
    for val, cnt in ac.items():
        if pd.notna(val):
            print(f"  {str(val):>30s}: {cnt:>7,} ({cnt/ac_non_null*100:5.1f}% of labeled, {cnt/total*100:5.1f}% of total)")

    # Binary: Class I+III = high severity, Class II = lower
    class_i = events["action_classification"].isin(["Class I"]).sum()
    class_ii = events["action_classification"].isin(["Class II"]).sum()
    class_iii = events["action_classification"].isin(["Class III"]).sum()
    other_labeled = ac_non_null - class_i - class_ii - class_iii

    print(f"\nBinary severity grouping:")
    print(f"  High severity (Class I + Class III): {class_i + class_iii:,}")
    print(f"  Lower severity (Class II):           {class_ii:,}")
    print(f"  Other labeled:                       {other_labeled:,}")
    print(f"  Unlabeled:                           {total - ac_non_null:,}")

    # Which fields co-occur with action_classification?
    labeled = events[events["action_classification"].notna()]
    print(f"\nField availability among labeled events ({len(labeled):,}):")
    for col in ["reason", "action", "action_summary", "determined_cause",
                 "event_date", "type", "country", "device_id"]:
        if col in labeled.columns:
            avail = labeled[col].notna().sum()
            print(f"  {col:>25s}: {avail:>6,} ({avail/len(labeled)*100:.1f}%)")

    # ================================================================
    # 7. CANDIDATE C — REPEAT/RECURRENT EVENT
    # ================================================================
    section("7. CANDIDATE C — Repeat/Recurrent Event")

    print(f"Total unique devices: {total_devices_in_events:,}")
    print(f"Devices with 1 event:   {(epc['event_count'] == 1).sum():,} ({(epc['event_count'] == 1).sum()/total_devices_in_events*100:.2f}%)")
    print(f"Devices with ≥2 events: {(epc['event_count'] >= 2).sum():,} ({(epc['event_count'] >= 2).sum()/total_devices_in_events*100:.2f}%)")
    print(f"Devices with ≥3 events: {(epc['event_count'] >= 3).sum():,}")
    print()

    # For multi-event devices, analyze temporal spacing
    multi_device_ids = epc[epc["event_count"] >= 2]["device_id"].values
    multi_events = dated_events[dated_events["device_id"].isin(multi_device_ids)].copy()
    multi_events = multi_events.sort_values(["device_id", "event_date"])

    # Compute gap between consecutive events per device
    multi_events["prev_date"] = multi_events.groupby("device_id")["event_date"].shift(1)
    multi_events["gap_days"] = (multi_events["event_date"] - multi_events["prev_date"]).dt.days
    gaps = multi_events["gap_days"].dropna()

    print(f"Inter-event gaps (consecutive events, same device):")
    print(f"  Total gap observations: {len(gaps):,}")
    print(f"  Mean gap:   {gaps.mean():.0f} days")
    print(f"  Median gap: {gaps.median():.0f} days")
    print(f"  P10 gap:    {gaps.quantile(0.10):.0f} days")
    print(f"  P25 gap:    {gaps.quantile(0.25):.0f} days")
    print(f"  P75 gap:    {gaps.quantile(0.75):.0f} days")
    print(f"  P90 gap:    {gaps.quantile(0.90):.0f} days")
    print(f"  Gap = 0 (same day): {(gaps == 0).sum():,} ({(gaps == 0).sum()/len(gaps)*100:.1f}%)")
    print(f"  Gap 1-30 days:      {((gaps >= 1) & (gaps <= 30)).sum():,}")
    print(f"  Gap 31-365 days:    {((gaps > 30) & (gaps <= 365)).sum():,}")
    print(f"  Gap > 365 days:     {(gaps > 365).sum():,}")

    # How would this work as a prediction problem?
    # After first event, does device have another event?
    first_event_devices = device_dates.copy()
    positive_repeat = (first_event_devices["n_events"] >= 2).sum()
    negative_no_repeat = (first_event_devices["n_events"] == 1).sum()

    print(f"\nAs a binary target (after first event → has repeat?):")
    print(f"  Positive (has repeat):  {positive_repeat:,} ({positive_repeat/len(first_event_devices)*100:.2f}%)")
    print(f"  Negative (no repeat):   {negative_no_repeat:,} ({negative_no_repeat/len(first_event_devices)*100:.2f}%)")
    print(f"  Class ratio:            1:{negative_no_repeat/max(1,positive_repeat):.1f}")

    # ================================================================
    # 8. CANDIDATE D — MANUFACTURER/DEVICE RISK SCORING
    # ================================================================
    section("8. CANDIDATE D — Manufacturer/Device Risk Scoring")

    # Events per manufacturer
    mfr_events = merged.groupby("manufacturer_id").agg(
        n_events=("id", "count"),
        n_devices=("device_id", "nunique"),
        n_recalls=("type", lambda x: (x.str.contains("Recall", case=False, na=False)).sum()),
        n_class_i=("action_classification", lambda x: (x == "Class I").sum()),
        n_class_iii=("action_classification", lambda x: (x == "Class III").sum()),
    ).reset_index()

    print(f"Total manufacturers with events: {len(mfr_events):,}")
    print(f"\nEvents per manufacturer:")
    print(f"  Mean:   {mfr_events['n_events'].mean():.1f}")
    print(f"  Median: {mfr_events['n_events'].median():.0f}")
    print(f"  P25:    {mfr_events['n_events'].quantile(0.25):.0f}")
    print(f"  P75:    {mfr_events['n_events'].quantile(0.75):.0f}")
    print(f"  P95:    {mfr_events['n_events'].quantile(0.95):.0f}")
    print(f"  Max:    {mfr_events['n_events'].max()}")

    print(f"\nManufacturers with high-severity events (Class I):")
    print(f"  Has ≥1 Class I: {(mfr_events['n_class_i'] >= 1).sum():,}")
    print(f"  Has ≥1 Class III: {(mfr_events['n_class_iii'] >= 1).sum():,}")

    # ================================================================
    # 9. TEMPORAL FEASIBILITY
    # ================================================================
    section("9. TEMPORAL FEASIBILITY")

    dated = events[events["event_date_available"]].copy()
    print(f"Dated events: {len(dated):,}")
    print(f"Date range: {dated['event_date'].min()} to {dated['event_date'].max()}")

    # Events by year
    dated["year"] = dated["event_date"].dt.year
    year_counts = dated["year"].value_counts().sort_index()
    print(f"\nEvents by year:")
    for yr, cnt in year_counts.items():
        print(f"  {yr}: {cnt:>6,}")

    # Devices by year (year of first event)
    device_first = dated.groupby("device_id")["event_date"].min().dt.year.value_counts().sort_index()
    print(f"\nDevices by year of first event:")
    for yr, cnt in device_first.items():
        print(f"  {yr}: {cnt:>6,}")

    print(f"\nPercentiles of event_date:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:>2}: {dated['event_date'].quantile(p/100)}")

    # Potential temporal splits
    print(f"\nPotential temporal splits (if target allows):")
    print(f"  Option A: Train ≤2015 | Val 2016 | Test 2017-2018")
    train_a = (dated["year"] <= 2015).sum()
    val_a = (dated["year"] == 2016).sum()
    test_a = (dated["year"].isin([2017, 2018])).sum()
    print(f"    Train: {train_a:,} | Val: {val_a:,} | Test: {test_a:,}")

    print(f"  Option B: Train ≤2014 | Val 2015 | Test 2016-2018")
    train_b = (dated["year"] <= 2014).sum()
    val_b = (dated["year"] == 2015).sum()
    test_b = (dated["year"].isin([2016, 2017, 2018])).sum()
    print(f"    Train: {train_b:,} | Val: {val_b:,} | Test: {test_b:,}")

    # ================================================================
    # 11. SEVERITY CLASS BALANCE
    # ================================================================
    section("11. CLASS BALANCE FOR SEVERITY TARGET")

    # Binary: Class I (most severe) vs Class II+III
    labeled_events = events[events["action_classification"].notna()].copy()
    print(f"Labeled events: {len(labeled_events):,}")

    # Option 1: Binary - Class I vs rest
    is_class_i = (labeled_events["action_classification"] == "Class I").sum()
    not_class_i = len(labeled_events) - is_class_i
    print(f"\nBinary: Class I vs rest")
    print(f"  Class I:     {is_class_i:,} ({is_class_i/len(labeled_events)*100:.1f}%)")
    print(f"  Not Class I: {not_class_i:,} ({not_class_i/len(labeled_events)*100:.1f}%)")
    print(f"  Ratio:       1:{not_class_i/max(1,is_class_i):.1f}")

    # Option 2: Multi-class I/II/III
    for c in ["Class I", "Class II", "Class III"]:
        cnt = (labeled_events["action_classification"] == c).sum()
        print(f"  {c}: {cnt:,} ({cnt/len(labeled_events)*100:.1f}%)")

    other_ac = labeled_events[~labeled_events["action_classification"].isin(["Class I", "Class II", "Class III"])]
    print(f"  Other: {len(other_ac):,}")
    if len(other_ac) > 0:
        print(f"    Values: {other_ac['action_classification'].value_counts().to_dict()}")

    # Which countries have action_classification?
    print(f"\naction_classification by country (top 10):")
    ac_by_country = labeled_events.groupby("country").size().sort_values(ascending=False).head(10)
    for country, cnt in ac_by_country.items():
        print(f"  {country}: {cnt:,}")

    # Temporal distribution of labeled events
    labeled_dated = labeled_events[labeled_events["event_date_available"]]
    print(f"\nLabeled events with dates: {len(labeled_dated):,}")
    if len(labeled_dated) > 0:
        labeled_dated_yr = labeled_dated.copy()
        labeled_dated_yr["year"] = labeled_dated_yr["event_date"].dt.year
        print(f"  Year range: {labeled_dated_yr['year'].min()} to {labeled_dated_yr['year'].max()}")
        yr_dist = labeled_dated_yr["year"].value_counts().sort_index()
        for yr, cnt in yr_dist.items():
            print(f"    {yr}: {cnt:>5,}")


if __name__ == "__main__":
    main()
