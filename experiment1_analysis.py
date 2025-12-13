import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statistics
import math

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def parse_filename(filename):
    """ファイル名から実験条件を解析"""
    # 例: 20241210_190436_luminanceMixture_cameraSpeed4_fps5_c_trialNumber1.csv
    pattern = r'(\d{8}_\d{6})_(\w+)_cameraSpeed(\d+)_fps(\d+)_(\w+)_trialNumber(\d+)\.csv'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'timestamp': match.group(1),
            'condition': match.group(2),  # luminanceMixture, wobble, continuous
            'camera_speed': int(match.group(3)),
            'fps': int(match.group(4)),
            'participant': match.group(5),
            'trial_number': int(match.group(6))
        }
    return None

def load_experiment1_data(data_dir="public/ExperimentData1"):
    """実験1のデータを読み込み、整理"""
    
    if not os.path.exists(data_dir):
        print(f"データディレクトリが見つかりません: {data_dir}")
        return {}
    
    all_data = {}
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
            
        file_info = parse_filename(filename)
        if not file_info:
            continue
            
        file_path = os.path.join(data_dir, filename)
        
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(file_path)
            
            # データの基本情報を記録
            data_info = {
                'file_info': file_info,
                'data': df,
                'rows': len(df),
                'columns': list(df.columns)
            }
            
            # 被験者別にデータを整理
            participant = file_info['participant']
            condition = file_info['condition']
            trial = file_info['trial_number']
            
            if participant not in all_data:
                all_data[participant] = {}
            
            if condition not in all_data[participant]:
                all_data[participant][condition] = {}
            
            all_data[participant][condition][trial] = data_info
            
        except Exception as e:
            print(f"ファイル読み込みエラー {filename}: {e}")
    
    return all_data

def analyze_vection_response(data_dict):
    """Vection Responseデータの分析"""
    
    print("=== 実験1 Vection Response 分析 ===\n")
    
    # 被験者別、条件別の統計
    participant_stats = {}
    
    for participant, conditions in data_dict.items():
        print(f"被験者: {participant}")
        participant_stats[participant] = {}
        
        for condition, trials in conditions.items():
            print(f"  条件: {condition}")
            participant_stats[participant][condition] = {}
            
            all_vection_responses = []
            
            for trial_num, trial_data in trials.items():
                df = trial_data['data']
                
                # Vection Responseカラムを探す
                vection_col = None
                for col in df.columns:
                    if 'Vection' in col or 'vection' in col.lower():
                        vection_col = col
                        break
                
                if vection_col is None:
                    print(f"    Trial {trial_num}: Vection Responseカラムが見つかりません")
                    continue
                
                vection_values = df[vection_col].values
                non_zero_vection = vection_values[vection_values > 0]
                
                if len(non_zero_vection) > 0:
                    trial_stats = {
                        'mean': np.mean(non_zero_vection),
                        'median': np.median(non_zero_vection),
                        'std': np.std(non_zero_vection),
                        'max': np.max(non_zero_vection),
                        'count': len(non_zero_vection),
                        'total_frames': len(vection_values)
                    }
                    
                    print(f"    Trial {trial_num}: 平均={trial_stats['mean']:.3f}, "
                          f"中央値={trial_stats['median']:.3f}, "
                          f"標準偏差={trial_stats['std']:.3f}, "
                          f"最大値={trial_stats['max']:.3f}, "
                          f"反応回数={trial_stats['count']}")
                    
                    all_vection_responses.extend(non_zero_vection)
                else:
                    print(f"    Trial {trial_num}: Vection反応なし")
            
            # 条件全体の統計
            if all_vection_responses:
                condition_stats = {
                    'mean': np.mean(all_vection_responses),
                    'median': np.median(all_vection_responses),
                    'std': np.std(all_vection_responses),
                    'max': np.max(all_vection_responses),
                    'total_responses': len(all_vection_responses)
                }
                
                participant_stats[participant][condition] = condition_stats
                
                print(f"  {condition}全体: 平均={condition_stats['mean']:.3f}, "
                      f"中央値={condition_stats['median']:.3f}, "
                      f"標準偏差={condition_stats['std']:.3f}, "
                      f"総反応回数={condition_stats['total_responses']}")
            else:
                print(f"  {condition}: 全試行でVection反応なし")
        
        print()
    
    return participant_stats

def compare_conditions(participant_stats):
    """条件間の比較分析"""
    
    print("=== 条件間比較分析 ===\n")
    
    # 条件別の統計を集計
    condition_data = {}
    
    for participant, conditions in participant_stats.items():
        for condition, stats in conditions.items():
            if condition not in condition_data:
                condition_data[condition] = []
            condition_data[condition].append(stats['mean'])
    
    # 各条件の統計
    print("条件別平均Vection Response:")
    for condition, values in condition_data.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {condition}: 平均={mean_val:.3f} ± {std_val:.3f} (n={len(values)})")
    
    # 条件間の統計的比較
    conditions = list(condition_data.keys())
    if len(conditions) >= 2:
        print("\n条件間の統計的比較:")
        
        for i in range(len(conditions)):
            for j in range(i+1, len(conditions)):
                cond1, cond2 = conditions[i], conditions[j]
                
                if condition_data[cond1] and condition_data[cond2]:
                    # t検定
                    t_stat, p_value = stats.ttest_ind(condition_data[cond1], condition_data[cond2])
                    
                    print(f"  {cond1} vs {cond2}: t={t_stat:.3f}, p={p_value:.3f}")
                    
                    if p_value < 0.05:
                        print(f"    → 統計的有意差あり (p < 0.05)")
                    else:
                        print(f"    → 統計的有意差なし")

def create_visualizations(data_dict, participant_stats):
    """可視化の作成"""
    
    # 1. 被験者別、条件別のVection Response平均値
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('実験1: 被験者別・条件別Vection Response分析', fontsize=16)
    
    # データを整理
    participants = list(participant_stats.keys())
    conditions = ['luminanceMixture', 'wobble', 'continuous']
    
    # プロット用データ
    plot_data = []
    for participant in participants:
        for condition in conditions:
            if condition in participant_stats[participant]:
                plot_data.append({
                    'participant': participant,
                    'condition': condition,
                    'mean_vection': participant_stats[participant][condition]['mean']
                })
    
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        # 1. 条件別箱ひげ図
        ax1 = axes[0, 0]
        sns.boxplot(data=df_plot, x='condition', y='mean_vection', ax=ax1)
        ax1.set_title('条件別Vection Response分布')
        ax1.set_xlabel('実験条件')
        ax1.set_ylabel('平均Vection Response')
        
        # 2. 被験者別棒グラフ
        ax2 = axes[0, 1]
        pivot_data = df_plot.pivot(index='participant', columns='condition', values='mean_vection')
        pivot_data.plot(kind='bar', ax=ax2)
        ax2.set_title('被験者別・条件別Vection Response')
        ax2.set_xlabel('被験者')
        ax2.set_ylabel('平均Vection Response')
        ax2.legend(title='条件')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. ヒートマップ
        ax3 = axes[1, 0]
        heatmap_data = df_plot.pivot(index='participant', columns='condition', values='mean_vection')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('被験者×条件ヒートマップ')
        
        # 4. 条件別散布図
        ax4 = axes[1, 1]
        for condition in conditions:
            condition_data = df_plot[df_plot['condition'] == condition]['mean_vection']
            if len(condition_data) > 0:
                ax4.scatter([condition] * len(condition_data), condition_data, alpha=0.7, label=condition)
        ax4.set_title('条件別Vection Response散布図')
        ax4.set_xlabel('実験条件')
        ax4.set_ylabel('平均Vection Response')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('experiment1_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(participant_stats, data_dict):
    """分析レポートの生成"""
    
    report = []
    report.append("# 実験1 分析レポート")
    report.append("")
    report.append("## 実験概要")
    report.append("- **目的**: 時間的線形輝度混合における速度知覚の非線形性を調整法で定量的に検証")
    report.append("- **方法**: 心理物理的調整法により、上下映像の速度感が最も一致する条件を測定")
    report.append("- **被験者数**: " + str(len(participant_stats)))
    report.append("")
    
    # 被験者情報
    report.append("## 被験者情報")
    for participant in participant_stats.keys():
        report.append(f"- 被験者 {participant}")
    report.append("")
    
    # 条件別結果
    report.append("## 条件別結果")
    
    conditions = ['luminanceMixture', 'wobble', 'continuous']
    for condition in conditions:
        report.append(f"### {condition}")
        
        condition_data = []
        for participant, conditions_dict in participant_stats.items():
            if condition in conditions_dict:
                condition_data.append(conditions_dict[condition]['mean'])
        
        if condition_data:
            mean_val = np.mean(condition_data)
            std_val = np.std(condition_data)
            report.append(f"- 平均Vection Response: {mean_val:.3f} ± {std_val:.3f}")
            report.append(f"- 被験者数: {len(condition_data)}")
        else:
            report.append("- データなし")
        report.append("")
    
    # 統計的比較
    report.append("## 統計的比較")
    report.append("### 条件間比較")
    
    condition_data = {}
    for participant, conditions in participant_stats.items():
        for condition, stats in conditions.items():
            if condition not in condition_data:
                condition_data[condition] = []
            condition_data[condition].append(stats['mean'])
    
    for condition, values in condition_data.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            report.append(f"- {condition}: {mean_val:.3f} ± {std_val:.3f}")
    
    report.append("")
    report.append("## 結論")
    report.append("実験1では、時間的線形輝度混合における速度知覚の非線形性を調整法により定量的に検証した。")
    report.append("各条件におけるVection Responseの分析により、輝度混合比率と主観的速度知覚の関係を明らかにした。")
    
    return "\n".join(report)

def main():
    """メイン分析関数"""
    
    print("実験1データ分析を開始します...")
    
    # データ読み込み
    data_dict = load_experiment1_data()
    
    if not data_dict:
        print("データが見つかりませんでした。")
        return
    
    print(f"読み込み完了: {len(data_dict)}被験者のデータ")
    
    # Vection Response分析
    participant_stats = analyze_vection_response(data_dict)
    
    # 条件間比較
    compare_conditions(participant_stats)
    
    # 可視化
    create_visualizations(data_dict, participant_stats)
    
    # レポート生成
    report = generate_report(participant_stats, data_dict)
    
    # レポート保存
    with open('experiment1_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n分析完了！")
    print("- レポート: experiment1_analysis_report.md")
    print("- グラフ: experiment1_analysis.png")

if __name__ == "__main__":
    main()