"""Compare a physics curve and machine learning with measured wind power.

The same-timestamp wind measurements are used to estimate turbine output.
This is not a future weather forecast. The held-out period is never used to
fit either the physics coefficient or the machine-learning model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SOURCE_COLUMNS = ["Date/Time", "LV ActivePower (kW)", "Wind Speed (m/s)"]


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    if frame.shape[1] != 5 or any(name not in frame for name in SOURCE_COLUMNS):
        raise ValueError("Expected the five-column Kaggle T1.csv file")

    # The source file has inconsistent encoding for the degree symbol. The
    # numeric columns are identified by position after checking the others.
    frame.columns = ["time_text", "actual_kw", "wind_ms", "maker_curve", "direction_deg"]
    frame["time"] = pd.to_datetime(frame["time_text"], format="%d %m %Y %H:%M", errors="coerce")
    for name in ["actual_kw", "wind_ms", "maker_curve", "direction_deg"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    initial_rows = len(frame)
    frame = frame.dropna(subset=["time", "actual_kw", "wind_ms", "maker_curve", "direction_deg"])
    frame = frame.drop_duplicates(subset="time", keep="first")
    invalid = (frame["wind_ms"] < 0) | (frame["maker_curve"] < 0) | (frame["actual_kw"] < 0)
    invalid_count = int(invalid.sum())
    frame = frame.loc[~invalid].sort_values("time").reset_index(drop=True)
    if len(frame) < 1000:
        raise ValueError("Too few valid rows after basic cleaning")

    minutes = frame["time"].diff().dt.total_seconds().div(60)
    audit = {
        "source_rows": initial_rows,
        "valid_rows": len(frame),
        "removed_missing_or_duplicate_rows": initial_rows - len(frame) - invalid_count,
        "removed_negative_rows": invalid_count,
        "time_gaps_over_10_min": int((minutes > 10).sum()),
        "first_time": str(frame["time"].iloc[0]),
        "last_time": str(frame["time"].iloc[-1]),
    }
    return frame, audit


def features(frame: pd.DataFrame) -> pd.DataFrame:
    theta = np.deg2rad(frame["direction_deg"].to_numpy())
    return pd.DataFrame(
        {
            "wind_ms": frame["wind_ms"].to_numpy(),
            "direction_sin": np.sin(theta),
            "direction_cos": np.cos(theta),
        },
        index=frame.index,
    )


def fit_physics(train: pd.DataFrame) -> dict:
    """Fit the scale of P~v^3 to the maker curve, never to measured output."""
    rated_kw = float(train["maker_curve"].max())
    if rated_kw <= 0:
        raise ValueError("Manufacturer curve has no positive power")
    operating = train[train["maker_curve"] > 0]
    cut_in_ms = float(operating["wind_ms"].min())
    rising = operating[operating["maker_curve"] < 0.9 * rated_kw]
    x = rising["wind_ms"].to_numpy() ** 3
    y = rising["maker_curve"].to_numpy()
    k = float(np.dot(x, y) / np.dot(x, x))
    return {"k_kw_per_ms_cubed": k, "cut_in_ms": cut_in_ms, "rated_kw": rated_kw}


def physical_power(wind_ms: np.ndarray, parameters: dict) -> np.ndarray:
    wind_ms = np.asarray(wind_ms, dtype=float)
    power = np.minimum(parameters["k_kw_per_ms_cubed"] * wind_ms**3, parameters["rated_kw"])
    return np.where(wind_ms >= parameters["cut_in_ms"], power, 0.0)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae_kw": round(float(mean_absolute_error(actual, predicted)), 3),
        "rmse_kw": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
        "bias_kw": round(float(np.mean(predicted - actual)), 3),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def save_plots(prediction: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.2})

    # Figure 1: a short contiguous held-out window suitable for a demo.
    sample = prediction.iloc[:144]
    fig, ax = plt.subplots(figsize=(13, 5))
    for column, label, color in [
        ("actual_kw", "Measured", "#173f5f"),
        ("ml_kw", "Machine learning", "#ef6c35"),
        ("physics_kw", "Cubic physics", "#3c8d7a"),
        ("maker_kw", "Manufacturer curve", "#8b80b6"),
    ]:
        ax.plot(sample["time"], sample[column], label=label, color=color, linewidth=1.6)
    ax.set(xlabel="Time", ylabel="Power (kW)", title="Held-out measurements and estimates: first 24 hours")
    ax.legend(ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "01_time_comparison.png", dpi=180)
    plt.close(fig)

    # Figure 2: physical interpretation and real operating scatter.
    fig, ax = plt.subplots(figsize=(9, 6))
    subset = prediction.iloc[:: max(1, len(prediction) // 3000)]
    ax.scatter(subset["wind_ms"], subset["actual_kw"], s=6, alpha=0.16, label="Measured")
    order = np.argsort(prediction["wind_ms"].to_numpy())
    ax.plot(prediction["wind_ms"].to_numpy()[order], prediction["physics_kw"].to_numpy()[order],
            color="#3c8d7a", linewidth=2, label="Cubic physics")
    ax.plot(prediction["wind_ms"].to_numpy()[order], prediction["maker_kw"].to_numpy()[order],
            color="#8b80b6", linewidth=2, alpha=0.8, label="Manufacturer curve")
    ax.set(xlabel="Wind speed (m/s)", ylabel="Power (kW)", title="Power curve and measured spread")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_power_curve.png", dpi=180)
    plt.close(fig)

    # Figure 3: MAE by wind-speed regime, with every method on the same rows.
    bins = [0, 4, 8, 12, 16, 30]
    labels = ["0-4", "4-8", "8-12", "12-16", "16-30"]
    groups = pd.cut(prediction["wind_ms"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for label in labels:
        sub = prediction.loc[groups == label]
        if len(sub):
            rows.append([label, *[np.mean(np.abs(sub[col] - sub["actual_kw"])) for col in
                                   ["physics_kw", "maker_kw", "ml_kw"]]])
    bar = pd.DataFrame(rows, columns=["Wind speed (m/s)", "Physics", "Manufacturer", "ML"])
    ax = bar.plot(x="Wind speed (m/s)", y=["Physics", "Manufacturer", "ML"], kind="bar",
                  figsize=(9, 5), color=["#3c8d7a", "#8b80b6", "#ef6c35"])
    ax.set(ylabel="MAE (kW)", title="Error by wind-speed regime")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(out / "03_mae_by_wind_speed.png", dpi=180)
    plt.close(ax.figure)
    bar.to_csv(out / "mae_by_wind_speed.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/T1.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame, audit = load_data(args.data)
    split = int(len(frame) * 0.75)
    train, test = frame.iloc[:split], frame.iloc[split:]
    if train["time"].max() >= test["time"].min():
        raise AssertionError("Chronological split failed")

    physics = fit_physics(train)
    model = RandomForestRegressor(n_estimators=70, max_depth=12, min_samples_leaf=8,
                                  n_jobs=-1, random_state=42)
    x_train, x_test = features(train), features(test)
    model.fit(x_train, train["actual_kw"])

    prediction = pd.DataFrame({
        "time": test["time"].to_numpy(),
        "wind_ms": test["wind_ms"].to_numpy(),
        "actual_kw": test["actual_kw"].to_numpy(),
        "physics_kw": physical_power(test["wind_ms"].to_numpy(), physics),
        "maker_kw": test["maker_curve"].to_numpy(),
        "ml_kw": np.maximum(0, model.predict(x_test)),
    })
    actual = prediction["actual_kw"].to_numpy()
    metrics = {
        "data_audit": audit,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_end": str(train["time"].iloc[-1]),
        "test_start": str(test["time"].iloc[0]),
        "physics_parameters": physics,
        "feature_names": list(x_train.columns),
        "scores": {
            "physics": score(actual, prediction["physics_kw"].to_numpy()),
            "manufacturer": score(actual, prediction["maker_kw"].to_numpy()),
            "machine_learning": score(actual, prediction["ml_kw"].to_numpy()),
        },
    }
    prediction.to_csv(args.out / "predictions.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_plots(prediction, args.out)

    print("Train:", len(train), "rows through", metrics["train_end"])
    print("Test:", len(test), "rows from", metrics["test_start"])
    print("Physics coefficient k:", round(physics["k_kw_per_ms_cubed"], 4), "kW/(m/s)^3")
    for name, values in metrics["scores"].items():
        print(
            f"{name:>16}: MAE={values['mae_kw']:.1f} kW, "
            f"RMSE={values['rmse_kw']:.1f} kW, R2={values['r2']:.4f}"
        )
    print("Saved:", args.out.resolve())


if __name__ == "__main__":
    main()
