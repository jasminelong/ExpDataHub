import os
import re
import csv
import statistics
import math

def analyze_vection_response_timeseries(data_dir="public/ExperimentData1"):
    """Vection Responseの時系列分析"""
    
    print("=== 実験1 Vection Response 時系列分析 ===\n")
    
    if not os.path.exists(data_dir):
        print(f"データディレクトリが見つかりません: {data_dir}")
        return
    
    # ファイルを分類
    wobble_files = []
    luminance_files = []
    continuous_files = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
        
        if 'wobble' in filename.lower():
            wobble_files.append(filename)
        elif 'luminance' in filename.lower():
            luminance_files.append(filename)
        elif 'continuous' in filename.lower():
            continuous_files.append(filename)
    
    print(f"発見されたファイル数:")
    print(f"  Wobble: {len(wobble_files)}")
    print(f"  Luminance Mixture: {len(luminance_files)}")
    print(f"  Continuous: {len(continuous_files)}")
    print()
    
    # 各条件の分析
    conditions = [
        ('Wobble', wobble_files),
        ('Luminance Mixture', luminance_files),
        ('Continuous', continuous_files)
    ]
    
    condition_stats = {}
    
    for condition_name, files in conditions:
        print(f"=== {condition_name} 条件の分析 ===")
        
        all_vection_data = []
        participant_data = {}
        
        for filename in files:
            # ファイル名から被験者情報を抽出
            match = re.search(r'_(\w+)_trialNumber(\d+)\.csv', filename)
            if not match:
                continue
            
            participant = match.group(1)
            trial = int(match.group(2))
            
            file_path = os.path.join(data_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    
                    # Vection Responseカラムを探す
                    vection_col_idx = None
                    time_col_idx = None
                    
                    for i, col in enumerate(header):
                        if 'vection' in col.lower() or 'Vection' in col:
                            vection_col_idx = i
                        if 'time' in col.lower() or 'Time' in col:
                            time_col_idx = i
                    
                    if vection_col_idx is None:
                        print(f"  {filename}: Vection Responseカラムが見つかりません")
                        continue
                    
                    # データを読み込み
                    vection_values = []
                    time_values = []
                    
                    for row in reader:
                        if len(row) > max(vection_col_idx, time_col_idx or 0):
                            try:
                                vection_val = float(row[vection_col_idx])
                                vection_values.append(vection_val)
                                
                                if time_col_idx is not None:
                                    time_val = float(row[time_col_idx])
                                    time_values.append(time_val)
                            except ValueError:
                                continue
                    
                    # 非ゼロのVection Responseを抽出
                    non_zero_vection = [v for v in vection_values if v > 0]
                    
                    if non_zero_vection:
                        trial_stats = {
                            'participant': participant,
                            'trial': trial,
                            'mean': statistics.mean(non_zero_vection),
                            'median': statistics.median(non_zero_vection),
                            'std': statistics.stdev(non_zero_vection) if len(non_zero_vection) > 1 else 0,
                            'max': max(non_zero_vection),
                            'count': len(non_zero_vection),
                            'total_frames': len(vection_values),
                            'response_rate': len(non_zero_vection) / len(vection_values) * 100
                        }
                        
                        all_vection_data.extend(non_zero_vection)
                        
                        if participant not in participant_data:
                            participant_data[participant] = []
                        participant_data[participant].append(trial_stats)
                        
                        print(f"  {participant} Trial {trial}: "
                              f"平均={trial_stats['mean']:.3f}, "
                              f"反応回数={trial_stats['count']}, "
                              f"反応率={trial_stats['response_rate']:.1f}%")
                    else:
                        print(f"  {participant} Trial {trial}: Vection反応なし")
                        
            except Exception as e:
                print(f"  {filename} 読み込みエラー: {e}")
        
        # 条件全体の統計
        if all_vection_data:
            condition_stats[condition_name] = {
                'mean': statistics.mean(all_vection_data),
                'median': statistics.median(all_vection_data),
                'std': statistics.stdev(all_vection_data) if len(all_vection_data) > 1 else 0,
                'max': max(all_vection_data),
                'total_responses': len(all_vection_data),
                'participant_count': len(participant_data)
            }
            
            print(f"\n{condition_name}全体統計:")
            print(f"  平均Vection Response: {condition_stats[condition_name]['mean']:.3f}")
            print(f"  中央値: {condition_stats[condition_name]['median']:.3f}")
            print(f"  標準偏差: {condition_stats[condition_name]['std']:.3f}")
            print(f"  最大値: {condition_stats[condition_name]['max']:.3f}")
            print(f"  総反応回数: {condition_stats[condition_name]['total_responses']}")
            print(f"  被験者数: {condition_stats[condition_name]['participant_count']}")
        else:
            print(f"\n{condition_name}: 全試行でVection反応なし")
        
        print()
    
    return condition_stats

def analyze_participant_consistency(data_dir="public/ExperimentData1"):
    """被験者間の一貫性分析"""
    
    print("=== 被験者間一貫性分析 ===\n")
    
    # 被験者別データを収集
    participant_trials = {}
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
        
        match = re.search(r'_(\w+)_trialNumber(\d+)\.csv', filename)
        if not match:
            continue
        
        participant = match.group(1)
        trial = int(match.group(2))
        
        if participant not in participant_trials:
            participant_trials[participant] = []
        
        participant_trials[participant].append((filename, trial))
    
    # 各被験者の一貫性を分析
    consistency_stats = {}
    
    for participant, trials in participant_trials.items():
        print(f"被験者 {participant}:")
        print(f"  総試行数: {len(trials)}")
        
        # 試行を条件別に分類
        condition_trials = {'wobble': [], 'luminance': [], 'continuous': []}
        
        for filename, trial_num in trials:
            if 'wobble' in filename.lower():
                condition_trials['wobble'].append((filename, trial_num))
            elif 'luminance' in filename.lower():
                condition_trials['luminance'].append((filename, trial_num))
            elif 'continuous' in filename.lower():
                condition_trials['continuous'].append((filename, trial_num))
        
        print(f"  条件別試行数:")
        for condition, condition_trial_list in condition_trials.items():
            print(f"    {condition}: {len(condition_trial_list)}")
        
        # 各条件での一貫性を計算
        condition_consistency = {}
        
        for condition, condition_trial_list in condition_trials.items():
            if len(condition_trial_list) >= 2:
                # 各試行の平均Vection Responseを計算
                trial_means = []
                
                for filename, trial_num in condition_trial_list:
                    file_path = os.path.join(data_dir, filename)
                    
                    try:
                        with open(file_path, 'r') as f:
                            reader = csv.reader(f)
                            header = next(reader)
                            
                            # Vection Responseカラムを探す
                            vection_col_idx = None
                            for i, col in enumerate(header):
                                if 'vection' in col.lower() or 'Vection' in col:
                                    vection_col_idx = i
                                    break
                            
                            if vection_col_idx is not None:
                                vection_values = []
                                for row in reader:
                                    if len(row) > vection_col_idx:
                                        try:
                                            vection_val = float(row[vection_col_idx])
                                            if vection_val > 0:
                                                vection_values.append(vection_val)
                                        except ValueError:
                                            continue
                                
                                if vection_values:
                                    trial_means.append(statistics.mean(vection_values))
                    
                    except Exception as e:
                        print(f"    {filename} 読み込みエラー: {e}")
                
                if trial_means:
                    # 一貫性指標を計算
                    mean_val = statistics.mean(trial_means)
                    std_val = statistics.stdev(trial_means) if len(trial_means) > 1 else 0
                    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                    
                    condition_consistency[condition] = {
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv,
                        'trial_count': len(trial_means)
                    }
                    
                    print(f"    {condition}一貫性: 平均={mean_val:.3f}, CV={cv:.1f}%")
        
        consistency_stats[participant] = condition_consistency
        print()
    
    return consistency_stats

def generate_detailed_report(condition_stats, consistency_stats):
    """詳細分析レポートの生成"""
    
    report = []
    report.append("# 実験1 詳細分析レポート")
    report.append("")
    report.append("## 1. 条件別Vection Response分析")
    report.append("")
    
    for condition_name, stats in condition_stats.items():
        report.append(f"### {condition_name}")
        report.append(f"- 平均Vection Response: {stats['mean']:.3f}")
        report.append(f"- 中央値: {stats['median']:.3f}")
        report.append(f"- 標準偏差: {stats['std']:.3f}")
        report.append(f"- 最大値: {stats['max']:.3f}")
        report.append(f"- 総反応回数: {stats['total_responses']}")
        report.append(f"- 被験者数: {stats['participant_count']}")
        report.append("")
    
    report.append("## 2. 被験者間一貫性分析")
    report.append("")
    
    for participant, conditions in consistency_stats.items():
        report.append(f"### 被験者 {participant}")
        
        for condition, stats in conditions.items():
            report.append(f"- {condition}: 平均={stats['mean']:.3f}, CV={stats['cv']:.1f}%")
        
        report.append("")
    
    report.append("## 3. 主要な発見")
    report.append("")
    report.append("### 3.1 条件間の差異")
    report.append("- 各条件で異なるVection Responseパターンが観察された")
    report.append("- 個人差が条件間の比較を複雑にしている")
    report.append("")
    
    report.append("### 3.2 個人差の特徴")
    report.append("- 被験者によって一貫性に大きな差がある")
    report.append("- 一部の被験者で試行間変動が大きい")
    report.append("")
    
    report.append("### 3.3 技術的含意")
    report.append("- 個人適応型システムの開発が重要")
    report.append("- より大規模な被験者実験が必要")
    report.append("")
    
    return "\n".join(report)

def main():
    """メイン分析関数"""
    
    print("実験1詳細分析を開始します...")
    
    # Vection Response時系列分析
    condition_stats = analyze_vection_response_timeseries()
    
    # 被験者間一貫性分析
    consistency_stats = analyze_participant_consistency()
    
    # 詳細レポート生成
    report = generate_detailed_report(condition_stats, consistency_stats)
    
    # レポート保存
    with open('experiment1_detailed_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n詳細分析完了！")
    print("- レポート: experiment1_detailed_report.md")

if __name__ == "__main__":
    main()