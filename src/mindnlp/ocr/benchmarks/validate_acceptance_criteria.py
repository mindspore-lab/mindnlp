"""
Issue #2379 验收标准评估脚本
自动验证所有验收标准是否达标
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# 延迟导入，避免在只检查文件时加载整个模型
OCREvaluator = None
OCRMetrics = None
VLMOCREngine = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _import_evaluation_modules():
    """延迟导入评估模块"""
    global OCREvaluator, OCRMetrics, VLMOCREngine
    if OCREvaluator is None:
        try:
            from mindnlp.ocr.core.evaluate import OCREvaluator as _OCREvaluator, OCRMetrics as _OCRMetrics
            from mindnlp.ocr.models.qwen2vl import VLMOCREngine as _VLMOCREngine
            OCREvaluator = _OCREvaluator
            OCRMetrics = _OCRMetrics
            VLMOCREngine = _VLMOCREngine
        except ImportError as e:
            logger.warning(f"无法导入评估模块: {e}")
            logger.warning("评估功能将不可用，但文件检查功能仍然可用")


class AcceptanceCriteriaValidator:
    """验收标准验证器"""

    def __init__(self):
        self.criteria = {
            "cer_reduction": {
                "description": "LoRA微调在目标数据集上CER降低20%以上",
                "target": 0.20,  # 20%降低
                "passed": False,
                "actual": None
            },
            "table_accuracy": {
                "description": "表格识别精度提升至95%以上",
                "target": 0.95,
                "passed": False,
                "actual": None
            },
            "formula_accuracy": {
                "description": "公式识别精度提升至90%以上",
                "target": 0.90,
                "passed": False,
                "actual": None
            },
            "scripts_provided": {
                "description": "提供完整的训练和评估脚本",
                "passed": False,
                "files": []
            },
            "model_weights": {
                "description": "提供微调模型权重和配置文件",
                "passed": False,
                "files": []
            },
            "documentation": {
                "description": "提供详细的微调文档和最佳实践",
                "passed": False,
                "files": []
            }
        }

    def validate_cer_reduction(
        self,
        base_model_results: str,
        lora_model_results: str
    ) -> bool:
        """
        验证CER降低标准

        Args:
            base_model_results: 基础模型评估结果JSON文件
            lora_model_results: LoRA模型评估结果JSON文件
        """
        logger.info("🔍 验证CER降低标准...")

        try:
            with open(base_model_results, 'r', encoding='utf-8') as f:
                base_results = json.load(f)
            with open(lora_model_results, 'r', encoding='utf-8') as f:
                lora_results = json.load(f)

            base_cer = base_results['metrics']['cer']
            lora_cer = lora_results['metrics']['cer']

            reduction = (base_cer - lora_cer) / base_cer

            logger.info(f"  基础模型 CER: {base_cer:.4f}")
            logger.info(f"  LoRA模型 CER: {lora_cer:.4f}")
            logger.info(f"  CER降低: {reduction*100:.2f}%")

            self.criteria["cer_reduction"]["actual"] = reduction
            self.criteria["cer_reduction"]["passed"] = reduction >= self.criteria["cer_reduction"]["target"]

            if self.criteria["cer_reduction"]["passed"]:
                logger.info(f"  ✅ CER降低达标 (>= {self.criteria['cer_reduction']['target']*100:.0f}%)")
            else:
                logger.warning(f"  ❌ CER降低未达标 (需要 >= {self.criteria['cer_reduction']['target']*100:.0f}%)")

            return self.criteria["cer_reduction"]["passed"]
        except Exception as e:
            logger.error(f"  ❌ 验证CER降低时出错: {e}")
            return False

    def validate_task_accuracy(
        self,
        task_type: str,
        test_data: str,
        model_name: str,
        lora_path: Optional[str] = None,
        device: str = "cpu"
    ) -> bool:
        """
        验证特定任务的识别精度

        Args:
            task_type: 任务类型 (table/formula/handwriting)
            test_data: 测试数据JSON文件
            model_name: 模型名称
            lora_path: LoRA权重路径
            device: 设备
        """
        logger.info(f"🔍 验证{task_type}识别精度...")

        # 导入评估模块
        _import_evaluation_modules()
        if OCREvaluator is None or VLMOCREngine is None:
            logger.error("  ❌ 评估模块未安装或不可用")
            return False

        try:
            # 加载模型
            engine = VLMOCREngine(
                model_name=model_name,
                device=device,
                lora_path=lora_path
            )

            # 加载测试数据
            evaluator = OCREvaluator(engine)
            results = evaluator.evaluate_dataset(test_data)

            # 计算准确率 (1 - CER)
            accuracy = 1 - results['metrics']['cer']

            logger.info(f"  {task_type}识别准确率: {accuracy*100:.2f}%")

            criterion_key = f"{task_type}_accuracy"
            if criterion_key in self.criteria:
                self.criteria[criterion_key]["actual"] = accuracy
                self.criteria[criterion_key]["passed"] = accuracy >= self.criteria[criterion_key]["target"]

                if self.criteria[criterion_key]["passed"]:
                    logger.info(f"  ✅ {task_type}识别精度达标 (>= {self.criteria[criterion_key]['target']*100:.0f}%)")
                else:
                    logger.warning(f"  ❌ {task_type}识别精度未达标 (需要 >= {self.criteria[criterion_key]['target']*100:.0f}%)")

                return self.criteria[criterion_key]["passed"]

            return False
        except Exception as e:
            logger.error(f"  ❌ 验证{task_type}识别精度时出错: {e}")
            return False

    def validate_files_existence(self) -> bool:
        """验证必需文件是否存在"""
        logger.info("🔍 验证必需文件...")

        project_root = Path(__file__).parent.parent.parent

        # 检查训练和评估脚本
        required_scripts = [
            "src/mindnlp/ocr/finetune/train_lora.py",
            "src/mindnlp/ocr/finetune/dataset.py",
            "src/mindnlp/ocr/finetune/evaluate.py",
            "src/mindnlp/ocr/finetune/prepare_dataset.py",
        ]

        # 可选的评估脚本（如果存在会更好）
        optional_scripts = [
            "src/mindnlp/ocr/core/evaluate.py",
            "examples/ocr_eval/evaluate_model.py"
        ]

        scripts_found = []
        scripts_missing = []

        for script in required_scripts:
            script_path = project_root / script
            if script_path.exists():
                scripts_found.append(script)
            else:
                scripts_missing.append(script)

        optional_found = []
        for script in optional_scripts:
            script_path = project_root / script
            if script_path.exists():
                optional_found.append(script)

        scripts_found.extend(optional_found)

        self.criteria["scripts_provided"]["files"] = scripts_found
        self.criteria["scripts_provided"]["passed"] = len(scripts_missing) == 0

        if self.criteria["scripts_provided"]["passed"]:
            logger.info(f"  ✅ 必需脚本文件已提供 ({len(required_scripts)}/{len(required_scripts)})")
            if optional_found:
                logger.info(f"  ✅ 可选脚本文件: {len(optional_found)}个")
        else:
            logger.warning(f"  ⚠️  部分必需脚本文件缺失:")
            for missing in scripts_missing:
                logger.warning(f"    - {missing}")

        # 检查文档 - 使用更灵活的匹配
        required_docs = [
            "docs/ocr_finetuning_guide.md",  # 小写版本
            "docs/OCR_FINETUNING_GUIDE.md",  # 大写版本
        ]

        optional_docs = [
            "examples/ocr_eval/README.md",
            "docs/ocr_supplement_README.md"
        ]

        docs_found = []
        docs_missing = []

        # 至少要有一个微调指南
        guide_found = False
        for doc in required_docs:
            doc_path = project_root / doc
            if doc_path.exists():
                docs_found.append(doc)
                guide_found = True
                break

        if not guide_found:
            docs_missing.append("docs/ocr_finetuning_guide.md 或 OCR_FINETUNING_GUIDE.md")

        # 检查可选文档
        for doc in optional_docs:
            doc_path = project_root / doc
            if doc_path.exists():
                docs_found.append(doc)

        self.criteria["documentation"]["files"] = docs_found
        self.criteria["documentation"]["passed"] = guide_found

        if self.criteria["documentation"]["passed"]:
            logger.info(f"  ✅ 微调文档已提供")
            if len(docs_found) > 1:
                logger.info(f"  ✅ 额外文档: {len(docs_found)-1}个")
        else:
            logger.warning(f"  ⚠️  缺少微调文档")

        return self.criteria["scripts_provided"]["passed"] and self.criteria["documentation"]["passed"]

    def validate_model_weights(self, lora_path: str) -> bool:
        """验证模型权重是否存在"""
        logger.info("🔍 验证模型权重...")

        lora_path = Path(lora_path)

        if not lora_path.exists():
            logger.warning(f"  ❌ LoRA权重路径不存在: {lora_path}")
            self.criteria["model_weights"]["passed"] = False
            return False

        # 检查必需文件
        required_files = [
            "adapter_model.npz",
            "adapter_config.json"
        ]

        files_found = []
        files_missing = []

        for file in required_files:
            file_path = lora_path / file
            if file_path.exists():
                files_found.append(str(file_path))
            else:
                files_missing.append(file)

        self.criteria["model_weights"]["files"] = files_found
        self.criteria["model_weights"]["passed"] = len(files_missing) == 0

        if self.criteria["model_weights"]["passed"]:
            logger.info(f"  ✅ 模型权重文件完整")
        else:
            logger.warning(f"  ❌ 部分模型文件缺失:")
            for missing in files_missing:
                logger.warning(f"    - {missing}")

        return self.criteria["model_weights"]["passed"]

    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """生成验收报告"""
        logger.info("\n" + "="*60)
        logger.info("Issue #2379 验收标准评估报告")
        logger.info("="*60 + "\n")

        total_criteria = len(self.criteria)
        passed_criteria = sum(1 for c in self.criteria.values() if c["passed"])

        for key, criterion in self.criteria.items():
            status = "✅ 通过" if criterion["passed"] else "❌ 未通过"
            logger.info(f"{status} - {criterion['description']}")

            if criterion.get("actual") is not None:
                if isinstance(criterion["actual"], float):
                    logger.info(f"       实际值: {criterion['actual']*100:.2f}%, 目标值: {criterion['target']*100:.0f}%")

            if criterion.get("files"):
                logger.info(f"       文件数: {len(criterion['files'])}")

        logger.info("\n" + "="*60)
        logger.info(f"总体结果: {passed_criteria}/{total_criteria} 项达标 ({passed_criteria/total_criteria*100:.1f}%)")
        logger.info("="*60 + "\n")

        report = {
            "issue": "#2379",
            "timestamp": str(Path.cwd()),
            "total_criteria": total_criteria,
            "passed_criteria": passed_criteria,
            "pass_rate": passed_criteria / total_criteria,
            "criteria": self.criteria,
            "overall_status": "PASSED" if passed_criteria == total_criteria else "FAILED"
        }

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"📄 报告已保存到: {output_file}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Issue #2379 验收标准评估")

    # 基本参数
    parser.add_argument("--mode", type=str, default="files",
                        choices=["files", "cer", "table", "formula", "all"],
                        help="验证模式")

    # CER验证参数
    parser.add_argument("--base_results", type=str,
                        help="基础模型评估结果JSON")
    parser.add_argument("--lora_results", type=str,
                        help="LoRA模型评估结果JSON")

    # 任务精度验证参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="模型名称")
    parser.add_argument("--lora_path", type=str,
                        help="LoRA权重路径")
    parser.add_argument("--table_test_data", type=str,
                        help="表格识别测试数据")
    parser.add_argument("--formula_test_data", type=str,
                        help="公式识别测试数据")
    parser.add_argument("--device", type=str, default="cpu",
                        help="设备")

    # 输出参数
    parser.add_argument("--output", type=str, default="acceptance_report.json",
                        help="报告输出文件")

    args = parser.parse_args()

    validator = AcceptanceCriteriaValidator()

    # 根据模式执行验证
    if args.mode in ["files", "all"]:
        validator.validate_files_existence()
        if args.lora_path:
            validator.validate_model_weights(args.lora_path)

    if args.mode in ["cer", "all"]:
        if args.base_results and args.lora_results:
            validator.validate_cer_reduction(args.base_results, args.lora_results)
        else:
            logger.warning("⚠️  跳过CER验证：缺少评估结果文件")

    if args.mode in ["table", "all"]:
        if args.table_test_data and args.lora_path:
            validator.validate_task_accuracy(
                "table",
                args.table_test_data,
                args.model_name,
                args.lora_path,
                args.device
            )
        else:
            logger.warning("⚠️  跳过表格识别验证：缺少测试数据或模型")

    if args.mode in ["formula", "all"]:
        if args.formula_test_data and args.lora_path:
            validator.validate_task_accuracy(
                "formula",
                args.formula_test_data,
                args.model_name,
                args.lora_path,
                args.device
            )
        else:
            logger.warning("⚠️  跳过公式识别验证：缺少测试数据或模型")

    # 生成报告
    report = validator.generate_report(args.output)

    # 返回状态码
    sys.exit(0 if report["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
