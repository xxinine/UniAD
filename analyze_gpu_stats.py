#!/usr/bin/env python3
"""
GPU统计数据分析脚本
读取GPU统计CSV文件并计算各项指标的平均值
"""

import pandas as pd
import numpy as np
import re
import sys
import os


def clean_numeric_value(value):
    """清理数值，移除单位和特殊字符"""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        # 移除百分号、MiB、W等单位
        value = re.sub(r'[^\d.]', '', value)
        if value:
            return float(value)
    return np.nan


def analyze_gpu_stats(csv_file):
    """分析GPU统计数据"""
    
    if not os.path.exists(csv_file):
        print(f"错误: 文件 {csv_file} 不存在")
        return
    
    print(f"正在分析文件: {csv_file}")
    print("=" * 50)
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 清理列名（移除空格）
        df.columns = df.columns.str.strip()
        
        print(f"数据总行数: {len(df)}")
        print(f"数据时间范围: {df.iloc[0, 0]} 到 {df.iloc[-1, 0]}")
        print()
        
        # 处理各个数值列
        stats = {}
        
        # 温度 (°C)
        if 'temperature.gpu' in df.columns:
            temp_values = df['temperature.gpu'].apply(clean_numeric_value)
            temp_values = temp_values.dropna()
            if len(temp_values) > 0:
                stats['温度 (°C)'] = {
                    '平均值': temp_values.mean(),
                    '最小值': temp_values.min(),
                    '最大值': temp_values.max(),
                    '标准差': temp_values.std()
                }
        
        # GPU利用率 (%)
        util_col = [col for col in df.columns if 'utilization.gpu' in col]
        if util_col:
            util_values = df[util_col[0]].apply(clean_numeric_value)
            util_values = util_values.dropna()
            if len(util_values) > 0:
                stats['GPU利用率 (%)'] = {
                    '平均值': util_values.mean(),
                    '最小值': util_values.min(),
                    '最大值': util_values.max(),
                    '标准差': util_values.std()
                }
        
        # 显存使用量 (MiB)
        memory_col = [col for col in df.columns if 'memory.used' in col]
        if memory_col:
            memory_values = df[memory_col[0]].apply(clean_numeric_value)
            memory_values = memory_values.dropna()
            if len(memory_values) > 0:
                stats['显存使用 (MiB)'] = {
                    '平均值': memory_values.mean(),
                    '最小值': memory_values.min(),
                    '最大值': memory_values.max(),
                    '标准差': memory_values.std()
                }
        
        # 功耗 (W)
        power_col = [col for col in df.columns if 'power.draw' in col]
        if power_col:
            power_values = df[power_col[0]].apply(clean_numeric_value)
            power_values = power_values.dropna()
            if len(power_values) > 0:
                stats['功耗 (W)'] = {
                    '平均值': power_values.mean(),
                    '最小值': power_values.min(),
                    '最大值': power_values.max(),
                    '标准差': power_values.std()
                }
        
        # 打印统计结果
        print("GPU性能统计:")
        print("=" * 50)
        
        for metric, values in stats.items():
            print(f"\n{metric}:")
            print(f"  平均值: {values['平均值']:.2f}")
            print(f"  最小值: {values['最小值']:.2f}")
            print(f"  最大值: {values['最大值']:.2f}")
            print(f"  标准差: {values['标准差']:.2f}")
        
        # 生成简要报告
        print("\n" + "=" * 50)
        print("简要报告:")
        if '温度 (°C)' in stats:
            avg_temp = stats['温度 (°C)']['平均值']
            print(f"• 平均温度: {avg_temp:.1f}°C {'(正常)' if avg_temp < 80 else '(偏高)' if avg_temp < 85 else '(过热)'}")
        
        if 'GPU利用率 (%)' in stats:
            avg_util = stats['GPU利用率 (%)']['平均值']
            print(f"• 平均GPU利用率: {avg_util:.1f}% {'(利用充分)' if avg_util > 80 else '(利用一般)' if avg_util > 50 else '(利用较低)'}")
        
        if '显存使用 (MiB)' in stats:
            avg_memory = stats['显存使用 (MiB)']['平均值']
            print(f"• 平均显存使用: {avg_memory:.0f} MiB ({avg_memory/1024:.1f} GiB)")
        
        if '功耗 (W)' in stats:
            avg_power = stats['功耗 (W)']['平均值']
            print(f"• 平均功耗: {avg_power:.1f}W")
        
        return stats
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return None


def main():
    """主函数"""
    
    # 默认文件名
    default_file = "log/gpu_stats.csv"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = default_file
    
    # 如果文件不存在，尝试在当前目录查找类似的文件
    if not os.path.exists(csv_file):
        # 查找所有GPU统计文件
        gpu_files = [f for f in os.listdir('.') if f.startswith('gpu_stats.csv')]
        
        if gpu_files:
            print(f"找到以下GPU统计文件:")
            for i, f in enumerate(gpu_files):
                print(f"  {i+1}. {f}")
            
            if len(gpu_files) == 1:
                csv_file = gpu_files[0]
                print(f"自动选择: {csv_file}")
            else:
                try:
                    choice = int(input("请选择文件编号: ")) - 1
                    if 0 <= choice < len(gpu_files):
                        csv_file = gpu_files[choice]
                    else:
                        print("无效选择")
                        return
                except ValueError:
                    print("无效输入")
                    return
        else:
            print("未找到GPU统计文件")
            return
    
    # 分析统计数据
    analyze_gpu_stats(csv_file)


if __name__ == "__main__":
    main()