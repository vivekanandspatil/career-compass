import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

st.set_page_config(page_title="Student Career Helper | 22CS53", page_icon="🎓", layout="wide")

FEATURES = [
    "CIE_Marks", "Coding_Aptitude", "Math_Score",
    "Analytics_Skill", "Communication_Score"
]
CAREERS = [
    "Software Engineer", "Data Scientist",
    "Research Scholar", "Cloud Administrator"
]


def generate_dataset(n=100, seed=22):
    """Create reproducible synthetic records; labels are generated from transparent profiles."""
    rng = np.random.default_rng(seed)
    profiles = {
        "Software Engineer": [39, 88, 70, 68, 66],
        "Data Scientist": [41, 73, 91, 92, 69],
        "Research Scholar": [44, 65, 95, 86, 77],
        "Cloud Administrator": [36, 77, 65, 72, 80],
    }
    rows = []
    labels = list(profiles)
    for i in range(n):
        career = labels[i % len(labels)]
        center = np.array(profiles[career], dtype=float)
        noise = np.array([
            rng.normal(0, 7), rng.normal(0, 13), rng.normal(0, 13),
            rng.normal(0, 13), rng.normal(0, 13)
        ])
        values = np.clip(center + noise, [0, 0, 0, 0, 0], [50, 100, 100, 100, 100])
        row = {feature: round(float(value), 1) for feature, value in zip(FEATURES, values)}
        row["Past_Successful_Career"] = career
        row["Student_ID"] = f"STU-{i + 1:03d}"
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[["Student_ID"] + FEATURES + ["Past_Successful_Career"]]


def minmax_scale(train_x, query_x=None):
    """Scale features to [0,1] using training-data ranges; constant columns are safe."""
    mins = train_x.min(axis=0)
    ranges = train_x.max(axis=0) - mins
    ranges = ranges.mask(ranges == 0, 1)
    scaled_train = (train_x - mins) / ranges
    if query_x is None:
        return scaled_train
    return scaled_train, (query_x - mins) / ranges


def knn_predict(df, student, k=5):
    """Custom K-NN using NumPy Euclidean distances; returns neighbor table and vote."""
    x = df[FEATURES].astype(float)
    query = pd.DataFrame([student], columns=FEATURES, dtype=float)
    x_scaled, query_scaled = minmax_scale(x, query)
    differences = x_scaled.to_numpy() - query_scaled.to_numpy()[0]
    distances = np.sqrt(np.sum(differences ** 2, axis=1))
    neighbors = df.copy()
    neighbors["Euclidean_Distance"] = distances
    neighbors = neighbors.sort_values("Euclidean_Distance", ascending=True).head(k).copy()
    votes = Counter(neighbors["Past_Successful_Career"])
    top_count = max(votes.values())
    tied = sorted([career for career, count in votes.items() if count == top_count])
    prediction = tied[0]
    return prediction, neighbors, votes, x_scaled, query_scaled


def calculate_entropy(labels):
    """Shannon entropy in bits."""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(np.asarray(labels, dtype=str), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def best_split(data, features):
    """Find the numeric threshold with highest information gain."""
    parent_entropy = calculate_entropy(data["Past_Successful_Career"].to_numpy())
    best = None
    trace = []
    for feature in features:
        values = np.sort(data[feature].astype(float).unique())
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        # Limit candidate thresholds to unique quantile candidates for efficient, readable training.
        if len(thresholds) > 24:
            indices = np.unique(np.linspace(0, len(thresholds) - 1, 24).astype(int))
            thresholds = thresholds[indices]
        for threshold in thresholds:
            left = data[data[feature] <= threshold]
            right = data[data[feature] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            left_entropy = calculate_entropy(left["Past_Successful_Career"])
            right_entropy = calculate_entropy(right["Past_Successful_Career"])
            weighted = (len(left) / len(data)) * left_entropy + (len(right) / len(data)) * right_entropy
            gain = parent_entropy - weighted
            record = {
                "feature": feature, "threshold": float(threshold),
                "parent_entropy": parent_entropy,
                "left_n": len(left), "left_entropy": left_entropy,
                "right_n": len(right), "right_entropy": right_entropy,
                "weighted_entropy": weighted, "information_gain": gain
            }
            trace.append(record)
            if best is None or gain > best["information_gain"]:
                best = record
    return best, trace


def train_decision_tree(data, max_depth=4, min_samples_split=8):
    """Build a small custom decision tree with entropy/information-gain splits."""
    trace = []

    def build(node_data, features, depth):
        labels = node_data["Past_Successful_Career"].astype(str)
        majority = labels.value_counts().sort_values(ascending=False)
        majority_label = sorted(majority[majority == majority.iloc[0]].index)[0]
        entropy = calculate_entropy(labels)
        node = {"prediction": majority_label, "samples": len(node_data), "entropy": entropy}
        if (entropy == 0 or depth >= max_depth or len(node_data) < min_samples_split
                or not features):
            node["leaf"] = True
            return node
        split, split_trace = best_split(node_data, features)
        trace.append({
            "depth": depth, "samples": len(node_data), "entropy": entropy,
            "candidate_count": len(split_trace), "best_split": split,
            "top_candidates": sorted(split_trace, key=lambda item: item["information_gain"], reverse=True)[:5]
        })
        if split is None or split["information_gain"] <= 1e-12:
            node["leaf"] = True
            return node
        node.update({
            "leaf": False, "feature": split["feature"], "threshold": split["threshold"],
            "left": build(node_data[node_data[split["feature"]] <= split["threshold"]], features, depth + 1),
            "right": build(node_data[node_data[split["feature"]] > split["threshold"]], features, depth + 1)
        })
        return node

    tree = build(data, FEATURES, 0)
    return tree, trace


def tree_predict(tree, student, path=None):
    """Predict with a trained tree and optionally return the decision path."""
    path = [] if path is None else path
    node = tree
    while not node.get("leaf", True):
        feature, threshold = node["feature"], node["threshold"]
        value = float(student[feature])
        go_left = value <= threshold
        path.append({
            "feature": feature, "value": value, "operator": "<=" if go_left else ">",
            "threshold": threshold, "branch": "left" if go_left else "right"
        })
        node = node["left"] if go_left else node["right"]
    return node["prediction"], path, node


def rule_engine(student, target_career):
    """Deterministic forward-chaining IF-THEN rules for target-career skill gaps."""
    requirements = {
        "Software Engineer": {"Coding_Aptitude": 80, "Math_Score": 65, "Analytics_Skill": 65, "Communication_Score": 60},
        "Data Scientist": {"Math_Score": 82, "Analytics_Skill": 82, "Coding_Aptitude": 68, "Communication_Score": 60},
        "Research Scholar": {"Math_Score": 85, "Analytics_Skill": 78, "Communication_Score": 70, "CIE_Marks": 38},
        "Cloud Administrator": {"Coding_Aptitude": 65, "Analytics_Skill": 65, "Communication_Score": 75, "CIE_Marks": 32},
    }
    rules = []
    gaps = []
    fired_facts = set()
    for skill, required in requirements[target_career].items():
        actual = float(student[skill])
        if actual < required:
            gap = required - actual
            fact = f"{skill}_BELOW_TARGET"
            if fact not in fired_facts:
                fired_facts.add(fact)
                message = f"IF {skill} = {actual:g} < {required}, THEN recommend improving {skill} by at least {gap:.1f} points."
                rules.append({"rule": f"R{len(rules)+1}", "condition": f"{skill} < {required}", "action": f"Improve {skill}", "result": f"Gap: {gap:.1f} points", "trace": message})
                gaps.append({"Skill": skill, "Current Score": actual, "Target Threshold": required, "Gap": round(gap, 1)})
        else:
            rules.append({"rule": f"R{len(rules)+1}", "condition": f"{skill} >= {required}", "action": f"Maintain {skill}", "result": "Requirement met", "trace": f"IF {skill} = {actual:g} >= {required}, THEN maintain current performance."})
    return rules, pd.DataFrame(gaps, columns=["Skill", "Current Score", "Target Threshold", "Gap"])


def main():
    st.title("🎓 Student Career Helper")
    st.caption("22CS53 • Explainable hybrid AI/ML pipeline • NumPy + Pandas from-scratch algorithms")
    st.info("Educational prototype: the historical records are synthetic, and career predictions are illustrative—not validated career assessments.")

    with st.sidebar:
        st.header("Student Input Dashboard")
        st.write("Adjust your academic and skill scores.")
        cie = st.slider("CIE Marks (0–50)", 0, 50, 38)
        coding = st.slider("Coding Aptitude (0–100)", 0, 100, 75)
        math = st.slider("Math Score (0–100)", 0, 100, 78)
        analytics = st.slider("Analytics Skill (0–100)", 0, 100, 74)
        communication = st.slider("Communication Score (0–100)", 0, 100, 70)
        k = st.slider("K nearest peers", min_value=1, max_value=10, value=5)
        auditor = st.checkbox("Enable Mathematical Auditor Trace Mode", value=False)
        run = st.button("Analyze My Career Path", type="primary", use_container_width=True)

    df = generate_dataset()
    st.subheader("Synthetic Historical Dataset")
    left, right, third = st.columns(3)
    left.metric("Historical students", len(df))
    right.metric("Career classes", df["Past_Successful_Career"].nunique())
    third.metric("K-NN neighbors", k)
    with st.expander("Preview generated dataset"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    student = {
        "CIE_Marks": cie, "Coding_Aptitude": coding, "Math_Score": math,
        "Analytics_Skill": analytics, "Communication_Score": communication
    }

    if run or "analysis_result" not in st.session_state:
        knn_career, neighbors, votes, scaled_x, scaled_query = knn_predict(df, student, k)
        tree, tree_trace = train_decision_tree(df)
        tree_career, decision_path, leaf = tree_predict(tree, student)
        # The decision tree is the primary classification; K-NN is the peer-based companion estimate.
        target = tree_career
        rules, gaps = rule_engine(student, target)
        st.session_state["analysis_result"] = {
            "knn_career": knn_career, "neighbors": neighbors, "votes": votes,
            "tree": tree, "tree_trace": tree_trace, "tree_career": tree_career,
            "decision_path": decision_path, "leaf": leaf, "rules": rules, "gaps": gaps,
            "student": student, "scaled_x": scaled_x, "scaled_query": scaled_query
        }

    result = st.session_state["analysis_result"]
    # Recompute after every slider interaction so the display always matches current values.
    knn_career, neighbors, votes, scaled_x, scaled_query = knn_predict(df, student, k)
    tree, tree_trace = train_decision_tree(df)
    tree_career, decision_path, leaf = tree_predict(tree, student)
    rules, gaps = rule_engine(student, tree_career)

    st.divider()
    st.header("Career Analysis Results")
    a, b = st.columns(2)
    with a:
        st.subheader("Module V — Decision Tree")
        st.metric("Primary predicted career", tree_career)
        st.caption(f"Terminal leaf contains {leaf['samples']} training records; entropy = {leaf['entropy']:.4f} bits.")
    with b:
        st.subheader("Module IV — K-NN")
        st.metric("Nearest-peer majority", knn_career)
        st.write("Neighbor vote counts:", dict(votes))

    st.subheader("Nearest Historical Academic Peers")
    st.dataframe(neighbors[["Student_ID"] + FEATURES + ["Past_Successful_Career", "Euclidean_Distance"]].round(4), use_container_width=True, hide_index=True)

    st.subheader("Module II — Forward Reasoning Skill-Gap Recommendations")
    if gaps.empty:
        st.success(f"All configured thresholds for {tree_career} are met. Keep building projects and practical experience.")
    else:
        st.write(f"Target career: **{tree_career}**. These deterministic rules identify thresholds below the configured target.")
        st.dataframe(gaps, use_container_width=True, hide_index=True)
    with st.expander("Show all fired and evaluated IF–THEN rules", expanded=auditor):
        for rule in rules:
            st.write(f"**{rule['rule']}** — IF {rule['condition']} → THEN {rule['action']}: {rule['result']}")

    st.subheader("Visual Analytics — Student vs. Nearest Peers")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    positions = np.arange(len(FEATURES))
    ax.plot(positions, [student[f] for f in FEATURES], marker="o", linewidth=2, label="Active student")
    for _, peer in neighbors.iterrows():
        ax.plot(positions, [peer[f] for f in FEATURES], marker=".", alpha=0.35, linewidth=1, label=peer["Student_ID"])
    ax.set_xticks(positions)
    ax.set_xticklabels(FEATURES, rotation=18, ha="right")
    ax.set_ylabel("Score (raw scale)")
    ax.set_title("Academic and skill profile comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="small")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if auditor:
        st.divider()
        st.header("🧮 Mathematical Auditor Trace Mode")
        st.markdown("### 1. K-NN Euclidean distance calculation")
        st.code(
            "For each historical student x and active student q:\n"
            "d(x, q) = sqrt(sum_j ((x_j - q_j) / range_j)^2)\n"
            "Features are min-max scaled using historical-data minima and ranges.\n"
            f"Active student vector: {student}\n"
            f"Nearest-peer distances:\n{neighbors[['Student_ID', 'Past_Successful_Career', 'Euclidean_Distance']].to_string(index=False)}",
            language="text"
        )
        st.markdown("### 2. Decision Tree entropy and information gain")
        st.write("Entropy: H(S) = -Σ p(c) log₂ p(c). Information Gain = H(parent) − weighted child entropy.")
        st.write(f"Root entropy: {tree['entropy']:.6f} bits; root split: {tree.get('feature', 'leaf')} {('<=' + format(tree.get('threshold', 0), '.4f')) if not tree.get('leaf', True) else ''}")
        for idx, node_trace in enumerate(tree_trace, start=1):
            st.markdown(f"**Node {idx} — depth {node_trace['depth']}, n={node_trace['samples']}**")
            if node_trace["best_split"]:
                best = node_trace["best_split"]
                st.code(
                    f"H(parent) = {best['parent_entropy']:.6f}\n"
                    f"Split: {best['feature']} <= {best['threshold']:.4f}\n"
                    f"Left: n={best['left_n']}, H(left)={best['left_entropy']:.6f}\n"
                    f"Right: n={best['right_n']}, H(right)={best['right_entropy']:.6f}\n"
                    f"Weighted entropy = ({best['left_n']}/{node_trace['samples']})*{best['left_entropy']:.6f} + "
                    f"({best['right_n']}/{node_trace['samples']})*{best['right_entropy']:.6f} = {best['weighted_entropy']:.6f}\n"
                    f"Information gain = {best['parent_entropy']:.6f} - {best['weighted_entropy']:.6f} = {best['information_gain']:.6f}\n\n"
                    "Top candidate splits:\n" + pd.DataFrame(node_trace["top_candidates"])[["feature", "threshold", "weighted_entropy", "information_gain"]].to_string(index=False),
                    language="text"
                )
        st.markdown("### 3. Active student's decision path")
        st.write(f"Predicted career: **{tree_career}**")
        if decision_path:
            st.dataframe(pd.DataFrame(decision_path), use_container_width=True, hide_index=True)
        else:
            st.write("The root node is already a leaf.")
        st.markdown("### 4. Forward-chaining rule trace")
        st.code("\n".join(rule["trace"] for rule in rules), language="text")
        st.caption("Rules are deterministic and evaluate the configured thresholds for the decision-tree career. They are not learned from the synthetic labels.")


if __name__ == "__main__":
    main()
