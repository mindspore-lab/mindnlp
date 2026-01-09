"""
多场景OCR测试 - 测试表格、公式、手写体等场景

使用方法:
    pytest tests/ocr/test_multi_scenario.py
    python tests/ocr/test_multi_scenario.py --model_name "Qwen/Qwen2-VL-7B-Instruct" --scenario table
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import time
from datetime import datetime

# 添加mindnlp到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mindnlp.ocr.core.engine import VLMOCREngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiScenarioTester:
    """多场景OCR测试器"""
    
    SCENARIO_CONFIGS = {
        "table": {
            "name": "表格识别",
            "description": "测试结构化表格的识别准确率",
            "prompt": "请识别图像中的表格内容，保持表格结构",
            "target_accuracy": 0.95,  # Issue #2379要求: 表格识别精度提升至95%以上
            "metrics": ["structure_accuracy", "content_accuracy"]
        },
        "formula": {
            "name": "公式识别",
            "description": "测试数学公式、LaTeX表达式的识别",
            "prompt": "请识别图像中的数学公式，输出LaTeX格式",
            "target_accuracy": 0.90,  # Issue #2379要求: 公式识别精度提升至90%以上
            "metrics": ["latex_accuracy", "symbol_accuracy"]
        },
        "handwriting": {
            "name": "手写体识别",
            "description": "测试手写文字的识别能力",
            "prompt": "请识别图像中的手写文字",
            "target_accuracy": 0.85,
            "metrics": ["cer", "wer"]
        },
        "mixed": {
            "name": "混合场景",
            "description": "测试包含多种语言、格式混合的复杂文档",
            "prompt": "请识别图像中的所有内容，包括文字、数字、符号",
            "target_accuracy": 0.90,
            "metrics": ["overall_accuracy"]
        },
        "business_doc": {
            "name": "商业文档",
            "description": "测试营业执照、发票、合同等商业文档",
            "prompt": "请识别图像中的所有信息字段",
            "target_accuracy": 0.95,
            "metrics": ["field_extraction", "accuracy"]
        }
    }
    
    def __init__(self, model_name: str, lora_path: str = None, device: str = "npu:0"):
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = device
        self.engine = None
        self.results = {}
    
    def load_model(self):
        """加载OCR模型"""
        logger.info(f"🚀 正在加载模型: {self.model_name}")
        if self.lora_path:
            logger.info(f"🔧 使用LoRA权重: {self.lora_path}")
        
        self.engine = VLMOCREngine(
            model_name=self.model_name,
            lora_weights_path=self.lora_path,
            device=self.device
        )
        logger.info("✅ 模型加载完成")
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """计算字符错误率"""
        if not reference:
            return 1.0 if hypothesis else 0.0
        
        len_ref = len(reference)
        len_hyp = len(hypothesis)
        
        dp = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
        
        for i in range(len_ref + 1):
            dp[i][0] = i
        for j in range(len_hyp + 1):
            dp[0][j] = j
        
        for i in range(1, len_ref + 1):
            for j in range(1, len_hyp + 1):
                if reference[i-1] == hypothesis[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
        
        return dp[len_ref][len_hyp] / len_ref if len_ref > 0 else 0.0
    
    def test_table_recognition(self, test_data: List[Dict]) -> Dict:
        """测试表格识别"""
        logger.info("📊 开始测试表格识别...")
        
        results = {
            "scenario": "table",
            "total_samples": len(test_data),
            "successful": 0,
            "failed": 0,
            "average_accuracy": 0.0,
            "target_accuracy": self.SCENARIO_CONFIGS["table"]["target_accuracy"],
            "detailed_results": []
        }
        
        total_accuracy = 0.0
        
        for idx, sample in enumerate(test_data):
            try:
                image_path = sample["image_path"]
                ground_truth = sample["ground_truth"]
                
                # 执行OCR
                start_time = time.time()
                prediction = self.engine.inference(
                    image_path=image_path,
                    prompt=self.SCENARIO_CONFIGS["table"]["prompt"]
                )
                inference_time = time.time() - start_time
                
                # 计算准确率 (基于CER的补数)
                cer = self.calculate_cer(ground_truth, prediction)
                accuracy = 1.0 - cer
                total_accuracy += accuracy
                
                results["successful"] += 1
                results["detailed_results"].append({
                    "sample_id": idx,
                    "image_path": image_path,
                    "accuracy": accuracy,
                    "cer": cer,
                    "inference_time": inference_time,
                    "prediction_length": len(prediction)
                })
                
                logger.info(f"  样本 {idx}: 准确率={accuracy:.2%}, CER={cer:.4f}")
                
            except Exception as e:
                logger.error(f"❌ 表格样本 {idx} 测试失败: {str(e)}")
                results["failed"] += 1
        
        if results["successful"] > 0:
            results["average_accuracy"] = total_accuracy / results["successful"]
        
        # 检查是否达标
        meets_target = results["average_accuracy"] >= results["target_accuracy"]
        results["meets_target"] = meets_target
        
        logger.info(f"✅ 表格识别完成: 平均准确率={results['average_accuracy']:.2%} (目标: {results['target_accuracy']:.0%})")
        if meets_target:
            logger.info("✅ 达到Issue #2379要求!")
        else:
            logger.warning(f"⚠️  未达到要求，差距: {(results['target_accuracy'] - results['average_accuracy'])*100:.2f}%")
        
        return results
    
    def test_formula_recognition(self, test_data: List[Dict]) -> Dict:
        """测试公式识别"""
        logger.info("🔢 开始测试公式识别...")
        
        results = {
            "scenario": "formula",
            "total_samples": len(test_data),
            "successful": 0,
            "failed": 0,
            "average_accuracy": 0.0,
            "target_accuracy": self.SCENARIO_CONFIGS["formula"]["target_accuracy"],
            "detailed_results": []
        }
        
        total_accuracy = 0.0
        
        for idx, sample in enumerate(test_data):
            try:
                image_path = sample["image_path"]
                ground_truth = sample["ground_truth"]  # LaTeX格式
                
                # 执行OCR
                start_time = time.time()
                prediction = self.engine.inference(
                    image_path=image_path,
                    prompt=self.SCENARIO_CONFIGS["formula"]["prompt"]
                )
                inference_time = time.time() - start_time
                
                # 计算准确率
                cer = self.calculate_cer(ground_truth, prediction)
                accuracy = 1.0 - cer
                total_accuracy += accuracy
                
                results["successful"] += 1
                results["detailed_results"].append({
                    "sample_id": idx,
                    "image_path": image_path,
                    "accuracy": accuracy,
                    "cer": cer,
                    "inference_time": inference_time,
                    "ground_truth": ground_truth,
                    "prediction": prediction
                })
                
                logger.info(f"  样本 {idx}: 准确率={accuracy:.2%}, CER={cer:.4f}")
                
            except Exception as e:
                logger.error(f"❌ 公式样本 {idx} 测试失败: {str(e)}")
                results["failed"] += 1
        
        if results["successful"] > 0:
            results["average_accuracy"] = total_accuracy / results["successful"]
        
        meets_target = results["average_accuracy"] >= results["target_accuracy"]
        results["meets_target"] = meets_target
        
        logger.info(f"✅ 公式识别完成: 平均准确率={results['average_accuracy']:.2%} (目标: {results['target_accuracy']:.0%})")
        if meets_target:
            logger.info("✅ 达到Issue #2379要求!")
        else:
            logger.warning(f"⚠️  未达到要求，差距: {(results['target_accuracy'] - results['average_accuracy'])*100:.2f}%")
        
        return results
    
    def test_handwriting_recognition(self, test_data: List[Dict]) -> Dict:
        """测试手写体识别"""
        logger.info("✍️  开始测试手写体识别...")
        
        results = {
            "scenario": "handwriting",
            "total_samples": len(test_data),
            "successful": 0,
            "failed": 0,
            "average_cer": 0.0,
            "target_accuracy": self.SCENARIO_CONFIGS["handwriting"]["target_accuracy"],
            "detailed_results": []
        }
        
        total_cer = 0.0
        
        for idx, sample in enumerate(test_data):
            try:
                image_path = sample["image_path"]
                ground_truth = sample["ground_truth"]
                
                # 执行OCR
                start_time = time.time()
                prediction = self.engine.inference(
                    image_path=image_path,
                    prompt=self.SCENARIO_CONFIGS["handwriting"]["prompt"]
                )
                inference_time = time.time() - start_time
                
                # 计算CER
                cer = self.calculate_cer(ground_truth, prediction)
                total_cer += cer
                
                results["successful"] += 1
                results["detailed_results"].append({
                    "sample_id": idx,
                    "image_path": image_path,
                    "cer": cer,
                    "accuracy": 1.0 - cer,
                    "inference_time": inference_time
                })
                
                logger.info(f"  样本 {idx}: CER={cer:.4f}, 准确率={1.0-cer:.2%}")
                
            except Exception as e:
                logger.error(f"❌ 手写体样本 {idx} 测试失败: {str(e)}")
                results["failed"] += 1
        
        if results["successful"] > 0:
            results["average_cer"] = total_cer / results["successful"]
            results["average_accuracy"] = 1.0 - results["average_cer"]
        
        meets_target = results.get("average_accuracy", 0) >= results["target_accuracy"]
        results["meets_target"] = meets_target
        
        logger.info(f"✅ 手写体识别完成: 平均准确率={results.get('average_accuracy', 0):.2%}, CER={results['average_cer']:.4f}")
        
        return results
    
    def test_scenario(self, scenario: str, test_data: List[Dict]) -> Dict:
        """测试指定场景"""
        if scenario == "table":
            return self.test_table_recognition(test_data)
        elif scenario == "formula":
            return self.test_formula_recognition(test_data)
        elif scenario == "handwriting":
            return self.test_handwriting_recognition(test_data)
        else:
            raise ValueError(f"不支持的场景: {scenario}")
    
    def run_all_tests(self, test_data_dir: str) -> Dict:
        """运行所有场景测试"""
        logger.info("🚀 开始运行所有场景测试...")
        
        all_results = {
            "model_name": self.model_name,
            "lora_path": self.lora_path,
            "test_time": datetime.now().isoformat(),
            "scenarios": {}
        }
        
        # 测试每个场景
        for scenario_name in ["table", "formula", "handwriting"]:
            test_file = Path(test_data_dir) / f"{scenario_name}_test.json"
            
            if not test_file.exists():
                logger.warning(f"⚠️  测试文件不存在: {test_file}")
                continue
            
            # 加载测试数据
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"测试场景: {self.SCENARIO_CONFIGS[scenario_name]['name']}")
            logger.info(f"描述: {self.SCENARIO_CONFIGS[scenario_name]['description']}")
            logger.info(f"测试样本数: {len(test_data)}")
            logger.info(f"{'='*80}")
            
            # 运行测试
            results = self.test_scenario(scenario_name, test_data)
            all_results["scenarios"][scenario_name] = results
        
        return all_results
    
    def generate_report(self, results: Dict, output_file: str):
        """生成测试报告"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 多场景测试报告")
        logger.info(f"{'='*80}")
        logger.info(f"模型: {results['model_name']}")
        if results['lora_path']:
            logger.info(f"LoRA: {results['lora_path']}")
        logger.info(f"测试时间: {results['test_time']}")
        logger.info(f"{'='*80}\n")
        
        summary = []
        
        for scenario_name, scenario_results in results["scenarios"].items():
            config = self.SCENARIO_CONFIGS[scenario_name]
            
            logger.info(f"场景: {config['name']}")
            logger.info(f"  总样本数: {scenario_results['total_samples']}")
            logger.info(f"  成功: {scenario_results['successful']}")
            logger.info(f"  失败: {scenario_results['failed']}")
            
            if "average_accuracy" in scenario_results:
                accuracy = scenario_results["average_accuracy"]
                target = scenario_results["target_accuracy"]
                meets = scenario_results["meets_target"]
                
                logger.info(f"  平均准确率: {accuracy:.2%}")
                logger.info(f"  目标准确率: {target:.0%}")
                logger.info(f"  是否达标: {'✅ 是' if meets else '❌ 否'}")
                
                summary.append({
                    "scenario": config['name'],
                    "accuracy": accuracy,
                    "target": target,
                    "meets_target": meets
                })
            
            logger.info("")
        
        # 保存详细结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 详细报告已保存: {output_file}")
        
        # 打印汇总
        logger.info(f"\n{'='*80}")
        logger.info("📈 Issue #2379 达标情况汇总")
        logger.info(f"{'='*80}")
        
        for item in summary:
            status = "✅" if item["meets_target"] else "❌"
            logger.info(f"{status} {item['scenario']}: {item['accuracy']:.2%} (目标: {item['target']:.0%})")
        
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="多场景OCR测试脚本")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--scenario", type=str, default="all", 
                       choices=["all", "table", "formula", "handwriting"],
                       help="测试场景")
    parser.add_argument("--test_data_dir", type=str, required=True, 
                       help="测试数据目录 (包含table_test.json, formula_test.json等)")
    parser.add_argument("--device", type=str, default="npu:0", help="设备")
    parser.add_argument("--output", type=str, default=None, help="报告输出文件")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = MultiScenarioTester(
        model_name=args.model_name,
        lora_path=args.lora_path,
        device=args.device
    )
    
    # 加载模型
    tester.load_model()
    
    # 运行测试
    if args.scenario == "all":
        results = tester.run_all_tests(args.test_data_dir)
    else:
        test_file = Path(args.test_data_dir) / f"{args.scenario}_test.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = {
            "model_name": args.model_name,
            "lora_path": args.lora_path,
            "test_time": datetime.now().isoformat(),
            "scenarios": {
                args.scenario: tester.test_scenario(args.scenario, test_data)
            }
        }
    
    # 生成报告
    output_file = args.output or f"multi_scenario_report_{int(time.time())}.json"
    tester.generate_report(results, output_file)


if __name__ == "__main__":
    main()
