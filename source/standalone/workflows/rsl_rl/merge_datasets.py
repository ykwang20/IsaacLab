import numpy as np
import torch
from typing import Dict, List, Any

def recursive_merge(
    dict1: Dict[str, Any], 
    dict2: Dict[str, Any]
) -> Dict[str, Any]:
    """递归合并两个字典，处理嵌套结构和torch tensor"""
    merged = {}
    for key in dict1.keys():
        if isinstance(dict1[key], dict):
            # 如果是字典，递归合并
            merged[key] = recursive_merge(dict1[key], dict2[key])
        elif torch.is_tensor(dict1[key]):
            # 如果是tensor，沿第0维合并
            merged[key] = torch.cat((dict1[key], dict2[key]), dim=0)
        else:
            # 其他类型直接合并（可能需要根据实际情况调整）
            merged[key] = [dict1[key], dict2[key]]
    return merged

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载数据集，自动转换numpy数组为torch tensor"""
    data = np.load(file_path, allow_pickle=True)
    dataset = []
    for step_data in data:
        converted = {}
        for key, value in step_data.items():
            if isinstance(value, np.ndarray):
                converted[key] = torch.from_numpy(value)
            elif isinstance(value, dict):
                converted[key] = {
                    k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                converted[key] = value
        dataset.append(converted)
    return dataset

def merge_datasets(
    dataset1: List[Dict[str, Any]], 
    dataset2: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """合并两个数据集，处理嵌套结构"""
    assert len(dataset1) == len(dataset2), "数据集长度必须相同"
    return [
        recursive_merge(d1, d2) 
        for d1, d2 in zip(dataset1, dataset2)
    ]

def save_dataset(dataset: List[Dict[str, Any]], output_path: str):
    """保存数据集，自动转换torch tensor为numpy数组"""
    def convert(value):
        if torch.is_tensor(value):
            return value
        elif isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value
    
    np_data = []
    for step_data in dataset:
        np_data.append({k: convert(v) for k, v in step_data.items()})
    np.save(output_path, np_data, allow_pickle=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="合并两个包含嵌套结构的数据集")
    parser.add_argument("--dataset1", 
                       default='/home/legrobot/IsaacLab/episodes_states_sim_23dof_new_model_last.npy',
                       type=str, 
                       help="第一个数据集路径")
    parser.add_argument("--dataset2",
                       default='/home/legrobot/IsaacLab/episodes_states_sim_23dof_new_model50.npy', 
                       type=str, 
                       help="第二个数据集路径")
    parser.add_argument("--output", 
                       default='/home/legrobot/IsaacLab/episodes_states_sim_23dof_new_merged.npy',
                       type=str, 
                       help="合并后输出路径")
    
    args = parser.parse_args()
    
    print(f"加载 {args.dataset1}...")
    dataset1 = load_dataset(args.dataset1)
    
    print(f"加载 {args.dataset2}...")
    dataset2 = load_dataset(args.dataset2)
    
    print("合并数据集...")
    merged_dataset = merge_datasets(dataset1, dataset2)
    
    print(f"保存到 {args.output}...")
    save_dataset(merged_dataset, args.output)
    
    print("合并完成！")