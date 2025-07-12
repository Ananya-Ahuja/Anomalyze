# Firewall Behavioral Baselining Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from parser import parse_firewall_log

# Configuration Parameters
REQUIRED_COLUMNS = [
    'timestamp', 'src_ip', 'dst_ip', 'protocol', 'action',
    'firewall_policy_name', 'segment_name', 'src_port', 'dst_port',
    'reason', 'bytes_sent', 'bytes_received', 'duration_secs'
]

BASELINE_STATS = {
    'bytes_sent': {'mean': 5000, 'std': 2000},
    'duration_secs': {'mean': 300, 'std': 100}
}
KNOWN_BENIGN_OUTLIERS = set()
NEW_ASN_SET = set()
RARE_PROTOCOLS = set()
RARE_PORTS = set()
XGB_CLF = None  # Replace with your trained model if available

# Utility Functions
def validate_and_parse_file(uploaded_file, required_columns, job_name="Analysis Job"):
    if uploaded_file is None:
        st.error(f"Please upload a file for {job_name}.")
        st.stop()
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    df = parse_firewall_log(df)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    return df

def phase1_if_kde_sensitive(df_agg_metrics, if_contamination=0.2, kde_bandwidth=0.4, kde_percentile=20):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_agg_metrics)
    if_model = IsolationForest(contamination=if_contamination, random_state=42)
    if_model.fit(X_scaled)
    if_scores = -if_model.decision_function(X_scaled)
    if_anomaly_flags = np.where(if_model.predict(X_scaled) == -1, 1, 0)
    kde_model = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth)
    kde_model.fit(X_scaled)
    kde_scores = np.exp(kde_model.score_samples(X_scaled))
    kde_threshold = np.percentile(kde_scores, kde_percentile)
    kde_anomaly_flags = (kde_scores <= kde_threshold).astype(int)
    return {
        'if_scores': if_scores,
        'kde_scores': kde_scores,
        'if_anomaly_flags': if_anomaly_flags,
        'kde_anomaly_flags': kde_anomaly_flags
    }

def phase2_rule_engine(df, baseline_stats, known_benign_outliers=None,
                       drift_std_threshold=3, new_asn_set=None,
                       rare_protocols=None, rare_ports=None,
                       deny_rate_threshold=0.2, large_transfer_threshold=None):
    df = df.copy()
    for col in baseline_stats:
        mean, std = baseline_stats[col]['mean'], baseline_stats[col]['std']
        drift_col = f'{col}_drift'
        df[drift_col] = np.abs(df[col] - mean) / (std + 1e-6)
        df[f'{col}_drift_flag'] = (df[drift_col] > drift_std_threshold).astype(int)
    df['profile_drift'] = df[[f'{col}_drift_flag' for col in baseline_stats]].max(axis=1)
    df['new_asn_flag'] = ~df['asn'].isin(new_asn_set) if 'asn' in df.columns and new_asn_set else 0
    df['rare_protocol_flag'] = df['protocol'].isin(rare_protocols).astype(int) if rare_protocols and 'protocol' in df.columns else 0
    df['rare_port_flag'] = df['dstPort'].isin(rare_ports).astype(int) if rare_ports and 'dstPort' in df.columns else 0
    df['deny_flag'] = (df['action'] == 'Deny').astype(int) if 'action' in df.columns else 0
    if large_transfer_threshold is None and 'bytes_sent' in df.columns:
        large_transfer_threshold = df['bytes_sent'].quantile(0.99)
    df['large_transfer_flag'] = (df['bytes_sent'] > large_transfer_threshold).astype(int) if 'bytes_sent' in df.columns else 0
    if known_benign_outliers:
        df['known_benign_flag'] = df.apply(
            lambda row: (row['srcIp'], row['dstIp'], row['protocol'], row['dstPort']) in known_benign_outliers,
            axis=1
        ).astype(int)
    else:
        df['known_benign_flag'] = 0
    df['phase2_alert'] = (
        (df['profile_drift'] == 1) |
        (df['new_asn_flag'] == 1) |
        (df['rare_protocol_flag'] == 1) |
        (df['rare_port_flag'] == 1) |
        (df['deny_flag'] == 1) |
        (df['large_transfer_flag'] == 1)
    ) & (df['known_benign_flag'] == 0)
    df['phase2_severity'] = np.where(
        df['profile_drift'] == 1, 'Critical',
        np.where(
            (df['new_asn_flag'] == 1) | (df['deny_flag'] == 1) | (df['large_transfer_flag'] == 1),
            'Medium',
            'Low'
        )
    )
    return df

def full_pipeline(
    df,
    baseline_stats,
    known_benign_outliers,
    new_asn_set,
    rare_protocols,
    rare_ports,
    xgb_clf=None
):
    categorical_cols = [
    'protocol', 'action', 'firewall_policy_name',
    'segment_name', 'reason'
    ]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    df_numeric = df_encoded.select_dtypes(include=[np.number])
    features = df_numeric.drop(columns=['isAnomaly'], errors='ignore')
    features_clean = features.dropna()
    phase1_results = phase1_if_kde_sensitive(features_clean)
    df_phase1 = df.loc[features_clean.index].copy()
    df_phase1['if_scores'] = phase1_results['if_scores']
    df_phase1['kde_scores'] = phase1_results['kde_scores']
    df_phase1['if_anomaly_flags'] = phase1_results['if_anomaly_flags']
    df_phase1['kde_anomaly_flags'] = phase1_results['kde_anomaly_flags']
    df_phase2 = phase2_rule_engine(
        df_phase1,
        baseline_stats,
        known_benign_outliers,
        3,
        new_asn_set,
        rare_protocols,
        rare_ports
    )
    severity_map = {'Low': 0, 'Medium': 1, 'Critical': 2}
    df_phase2['phase2_severity_num'] = df_phase2['phase2_severity'].map(severity_map)
    feature_cols = [
        'bytes_sent', 'duration_secs',
        'if_scores', 'kde_scores', 'if_anomaly_flags', 'kde_anomaly_flags',
        'profile_drift', 'new_asn_flag', 'rare_protocol_flag', 'rare_port_flag',
        'deny_flag', 'large_transfer_flag', 'phase2_alert', 'phase2_severity_num'
    ]
    phase3_features = df_phase2[feature_cols].dropna()
    if xgb_clf is not None:
        preds = xgb_clf.predict(phase3_features)
        df_phase2.loc[phase3_features.index, 'meta_prediction'] = preds
    return df_phase2

# Streamlit App Main Function
def workflow():
    st.title("Firewall Behavioral Baselining Test App")
    uploaded_file = st.file_uploader("Upload your firewall log CSV file", type=["csv"])
    if uploaded_file:
        df = validate_and_parse_file(uploaded_file, REQUIRED_COLUMNS)
        results_df = full_pipeline(
            df,
            BASELINE_STATS,
            KNOWN_BENIGN_OUTLIERS,
            NEW_ASN_SET,
            RARE_PROTOCOLS,
            RARE_PORTS,
            XGB_CLF
        )
        st.success("Analysis complete!")
        st.dataframe(results_df, use_container_width=True)
        st.subheader("Baseline vs. Current Comparison")
        for col in BASELINE_STATS:
            chart_df = pd.DataFrame({
                "Baseline Mean": [BASELINE_STATS[col]['mean']],
                "Current Mean": [results_df[col].mean()]
            }, index=[col])
            st.bar_chart(chart_df)
    else:
        st.info("Please upload a CSV file to begin analysis.")