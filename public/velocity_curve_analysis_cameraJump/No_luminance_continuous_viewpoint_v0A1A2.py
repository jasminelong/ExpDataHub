import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_params_from_file(p: Path):
    """从文件提取 V0, A1, A2（兼容 StepNumber/Amplitude/Velocity 列），返回 dict"""
    params = {'V0': np.nan, 'A1': np.nan, 'A2': np.nan}
    try:
        if not p.exists():
            return params
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "StepNumber" in df.columns:
            # V0: StepNumber == 0, prefer Velocity 列
            if "Velocity" in df.columns:
                s0 = df[df["StepNumber"] == 0]["Velocity"]
                if not s0.empty:
                    params['V0'] = float(s0.iloc[-1])
            elif "Amplitude" in df.columns:
                s0 = df[df["StepNumber"] == 0]["Amplitude"]
                if not s0.empty:
                    params['V0'] = float(s0.iloc[-1])
            # A1/A2 from Amplitude at steps 1 and 3 respectively
            if "Amplitude" in df.columns:
                s1 = df[df["StepNumber"] == 1]["Amplitude"]
                if not s1.empty:
                    params['A1'] = float(s1.iloc[-1])
                s3 = df[df["StepNumber"] == 3]["Amplitude"]
                if not s3.empty:
                    params['A2'] = float(s3.iloc[-1])
            else:
                # fallback column names
                for col in ("Amplitude1","A1","Amp","Amplitude"):
                    if col in df.columns and np.isnan(params['A1']):
                        try:
                            params['A1'] = float(df[col].iloc[-1])
                        except Exception:
                            pass
                for col in ("Amplitude2","A2"):
                    if col in df.columns and np.isnan(params['A2']):
                        try:
                            params['A2'] = float(df[col].iloc[-1])
                        except Exception:
                            pass
                if "Velocity" in df.columns and np.isnan(params['V0']):
                    try:
                        params['V0'] = float(df["Velocity"].iloc[-1])
                    except Exception:
                        pass
        else:
            # no StepNumber: try direct column names
            for key, col in (('V0','Velocity'), ('A1','Amplitude1'), ('A2','Amplitude2')):
                if col in df.columns:
                    try:
                        params[key] = float(df[col].iloc[-1])
                    except Exception:
                        pass
    except Exception:
        pass
    return params

def analyze_kk_experiment33(data_dir=r"D:\vectionProject\public\ExperimentData33"):
    """查找 KK 的四种条件文件，提取 V0/A1/A2 并画条形图比较（保存并打印均值）
    另外将用户提供的 extra_paths1 三个 LinearOnly 文件作为额外一组一并比较。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print("data_dir not found:", data_path)
        return

    csv_files = list(data_path.glob("*.csv"))
    kk_files = [p for p in csv_files if ("ParticipantName_KK" in p.name) or ("_KK_" in p.name)]
    # 额外的 three LinearOnly files（按用户提供路径）
    extra_paths1 = [
        Path(r"D:\vectionProject\public\BrightnessData\20250709_152809_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1_BrightnessBlendMode_LinearOnly.csv"),
        Path(r"D:\vectionProject\public\BrightnessData\20250709_151437_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_3_BrightnessBlendMode_LinearOnly.csv"),
        Path(r"D:\vectionProject\public\BrightnessData\20250709_154001_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_2_BrightnessBlendMode_LinearOnly.csv"),
    ]
    # 将 extra_paths1 中存在的文件加入文件列表（避免重复）
    for p in extra_paths1:
        if p.exists() and p not in kk_files:
            kk_files.append(p)

    if not kk_files:
        print("No KK files found in", data_path, "or extra_paths1.")
        return

    # 分类四种条件 + 额外 LinearOnly 组
    groups = {
        "V0_only": [],
        "V0_A1": [],
        "V0_A2": [],
        "V0_A1A2": [],
        "KK_LinearOnly": []
    }
    for p in kk_files:
        name = p.name
        if "V0_A1A2" in name:
            groups["V0_A1A2"].append(p)
        elif "V0_A1" in name and "V0_A1A2" not in name:
            groups["V0_A1"].append(p)
        elif "V0_A2" in name and "V0_A1A2" not in name:
            groups["V0_A2"].append(p)
        elif "V0_" in name and all(k not in name for k in ("V0_A1","V0_A2","V0_A1A2")):
            groups["V0_only"].append(p)
        # 如果文件名或来源属于额外 LinearOnly 路径集合，也放到 KK_LinearOnly 组
        if any(str(p.resolve()) == str(ep.resolve()) for ep in extra_paths1):
            groups["KK_LinearOnly"].append(p)

    # Report counts
    print("KK groups counts:")
    for k, lst in groups.items():
        print(f"  {k}: {len(lst)} files")

    # 保留每组前5次（若有）
    for k in groups:
        groups[k] = sorted(groups[k])[:5]

    summary = {}
    for k, files in groups.items():
        vals_V0, vals_A1, vals_A2 = [], [], []
        print(f"\n--- Group {k} ({len(files)} files) ---")
        for p in files:
            params = get_params_from_file(p)
            print(f"{p.name} -> V0={params['V0']}, A1={params['A1']}, A2={params['A2']}")
            if not np.isnan(params['V0']):
                vals_V0.append(params['V0'])
            if not np.isnan(params['A1']):
                vals_A1.append(params['A1'])
            if not np.isnan(params['A2']):
                vals_A2.append(params['A2'])
        summary[k] = {
            'V0': np.array(vals_V0, dtype=float),
            'A1': np.array(vals_A1, dtype=float),
            'A2': np.array(vals_A2, dtype=float)
        }

    metrics = ['V0','A1','A2']
    groups_order = ["V0_only","V0_A1","V0_A2","V0_A1A2","KK_LinearOnly"]
    means = {m: [] for m in metrics}
    stds = {m: [] for m in metrics}
    for k in groups_order:
        for m in metrics:
            arr = summary[k][m]
            if arr.size == 0:
                means[m].append(np.nan)
                stds[m].append(np.nan)
            else:
                means[m].append(float(np.nanmean(arr)))
                stds[m].append(float(np.nanstd(arr, ddof=0)))

    print("\n=== Summary means (V0, A1, A2) per group ===")
    for k in groups_order:
        print(f"{k}:")
        for m in metrics:
            idx = groups_order.index(k)
            val = means[m][idx]
            sd = stds[m][idx]
            if np.isnan(val):
                print(f"  {m}: n=0")
            else:
                print(f"  {m}: mean={val:.4f}, std={sd:.4f}")

    # 画图：三种度量并排对比（包含额外 LinearOnly 组）
    labels = groups_order
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10,5))
    for i, m in enumerate(metrics):
        vals = means[m]
        errs = []
        for j in range(len(labels)):
            arr = summary[labels[j]][m]
            errs.append((np.nanstd(arr, ddof=0)/np.sqrt(arr.size)) if arr.size>0 else 0.0)
        ax.bar(x + (i-1)*width, vals, width, yerr=errs, capsize=4, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("value")
    ax.set_title("KK: V0 / A1 / A2 comparison across conditions (including LinearOnly)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    out_png = Path.cwd() / "KK_conditions_V0_A1_A2_with_LinearOnly.png"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print("\nSaved figure to", out_png.resolve(), " (exists=", out_png.exists(), ")")
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
    plt.close(fig)

if __name__ == "__main__":
    analyze_kk_experiment33()
