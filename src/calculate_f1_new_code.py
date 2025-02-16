import json
import os
from tqdm import tqdm
from evaluation_code.evaluator import *

def calculate_f1(output_dir, data_dir):
    EvaluatorDict = {
        'RE':EvaluatorRE,
        'EE':EvaluatorEvent,
        'NER':EvaluatorNER,
        'EET':EvaluatorEET,
        'EEA':EvaluatorEEA
    }
    task_dict = dict()      # str -> dict
    task_path = os.path.join(output_dir, 'generated_predictions.jsonl')
    report_dir_root = os.path.join(output_dir, 'report')
    result = []
    with open(task_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            result.append(data)   
    with open(data_dir, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        assert len(datas) == len(result)
        for index, data in tqdm(enumerate(datas)):
            task_name = data['Task']
            dataset_name = data['Dataset']
            if task_name not in task_dict:
                task_dict[task_name] = dict()
            if dataset_name not in task_dict[task_name]:
                task_dict[task_name][dataset_name] = EvaluatorDict[task_name]()
            assert result[index]['label'] in data['Instance']['label']
            task_dict[task_name][dataset_name].add(data, result[index]['predict'])

    # export report
    if not os.path.exists(report_dir_root):
        os.mkdir(report_dir_root)

    # export tsv
    for task_name, eval_dict in task_dict.items():
        print('\n'+'-'*16+task_name+'-'*16+'\n')
        rows = []
        scores = []
        report_dir = os.path.join(report_dir_root, task_name)
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        for dataset_name, evaluator in eval_dict.items():
            evaluator.dump_audit_report(os.path.join(report_dir, dataset_name+'.json'))
            rows.append((dataset_name, evaluator.get_metric()))
            scores.append(evaluator.get_metric())
        rows = sorted(rows, key=lambda x: x[0].lower())
        if len(scores) == 0:
            continue
        rows.append(('Average', sum(scores)/len(scores)))
        with open(os.path.join(report_dir_root, 'report_%s.tsv'%task_name), 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(f'{row[0]}\t{row[1]}\n')
                print('%48s\t%g'%row)

if __name__ == '__main__':
    root = 'expr/code_sft_new2/CodeLlama-13B/qlora8/evaluation_small'
    os.environ['RANDOM_RECORD'] = '1'   # 是否开启随机记录
    os.environ['EXPORT_IMG'] = '0'      # 是否导出混淆矩阵图片
    calculate_f1(root, "processed/code_sft_new/test.json")