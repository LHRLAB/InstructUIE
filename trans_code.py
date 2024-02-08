# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""InstructUIE Dataset."""

import json
import os
import random
import time

import datasets
from hashlib import md5

logger = datasets.logging.get_logger(__name__)
TASK_CONFIG_FILES = {"train": "train_tasks.json",
                     "dev": "dev_tasks.json", "test": "test_tasks.json"}
INSTRUCTION_STRATEGIES = ['single', 'multiple']
ANSWER_PREFIX = "Answer:"
SINGLE_QUOTES_SUBSTITUTE = "#$%#"
AUX_PROB = 0.3


def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
        data_args.instruction_file + data_args.instruction_strategy + \
        str(data_args.max_num_instances_per_task) + \
        str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))

    return cache_path


def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError(
            '{} is not valid, please check the input path!'.format(path))


def save_ds(instances, file_name):
    with open(file_name, "w+", encoding='utf-8') as fi:
        json.dump(instances, fi, ensure_ascii=False, indent=2)


class UIEConfig():
    """
    Config dataset load procedure.

    Args:
        data_dir: task data dir, which contains the corresponding dataset dirs
        prompt_path: prompt json file, which saves task and its prompts map
        task_file: task config file, save training and testing split config, and sampling strategies.
         Support two sampling strategies: 'random' indicates random sampling, while 'full' means to return all samples.
        max_num_instances_per_task: max training sample size of each task
        max_num_instances_per_eval_task: max eval sample size of each task
    """

    def __init__(
            self,
            *args,
            data_dir="IE_INSTRUCTIONS",
            instruction_file="configs/instruction_config.json",
            instruction_strategy="single",
            task_config_dir="configs/multi_task_configs",
            num_examples=0,
            max_num_instances_per_task=10000,
            max_num_instances_per_eval_task=200,
            over_sampling=None,
            **kwargs
    ):
        # super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_examples = num_examples
        self.over_sampling = over_sampling
        self.instructions = self._parse_instruction(instruction_file)
        self.task_configs = self._parse_task_config(task_config_dir)
        self.instruction_strategy = instruction_strategy
        self.max_num_instances_per_task = max_num_instances_per_task
        self.max_num_instances_per_eval_task = max_num_instances_per_eval_task

    def _parse_instruction(self, instruction_file):
        """
        Instruction example:
        {
          "RE": [
            {"instruction_type": "zero-shot", "instruction": "Given a phrase that describes the relationship between
            two words, extract the words and the lexical relationship between them.
            The output format should be :[(word1, relation, word2)]. \n"},
          ],
          "NER": [
            {"instruction_type": "zero-shot", "instruction": "Please list all entity words in the text that
            fit the category.Output format is [(word1, type1), (word2, type2))]. \n"},
          ],
          "EE": [
            {"instruction_type": "zero-shot", "instruction": "Extract the event information in the text
            and return them in the event list. \n"}
          ]
        }
        """
        if not instruction_file:
            return None
        instructions = {"zero-shot": {}, "few-shot": {}}

        with open(instruction_file, 'r+') as f:
            origin_instructions = json.load(f)

        for task in origin_instructions:
            for task_instruction in origin_instructions[task]:
                instruct_type = task_instruction["instruction_type"]
                if instruct_type == "zero-shot":
                    instructions['zero-shot'][task] = instructions['zero-shot'].get(task, [
                    ])
                    instructions['zero-shot'][task].append(
                        task_instruction["instruction"])
                elif instruct_type == "few-shot":
                    instructions['few-shot'][task] = instructions['few-shot'].get(task, [
                    ])
                    instructions['few-shot'][task].append(
                        task_instruction["instruction"])
                else:
                    raise ValueError("Invalid instruction type {}, please check your instruction file {}"
                                     .format(instruct_type, instruction_file))
        return instructions

    def _parse_task_config(self, task_config_dir):
        """
        Task config file example:
            {
              "RE": [
                {"sampling strategy": "random", "dataset name": "conll04"}
              ],
              "NER": [
                {"sampling strategy": "random", "dataset name": "ACE05_coarse-grained"},
                {"sampling strategy": "full", "dataset name": "conll2003"}
              ],
              "EE": [
                {"sampling strategy": "random", "dataset name": "GENIA"}
              ]
            }
        """
        if not task_config_dir:
            return None

        task_configs = {}
        for task, file_name in TASK_CONFIG_FILES.items():
            task_config_file = os.path.join(task_config_dir, file_name)

            if not os.path.exists(task_config_file):
                raise ValueError('Please check {} config, {} not exists!'.format(
                    task, task_config_file))

            with open(task_config_file, 'r+') as f:
                task_configs[task] = json.loads(f.read())

        return task_configs

    # TODO, few-shot, 需要 load 的时候就将值存好，放在 "Examples" 里面
    """InstructUIE Dataset."""


VERSION = datasets.Version("2.0.0")
BUILDER_CONFIG_CLASS = UIEConfig
BUILDER_CONFIGS = [
    UIEConfig(name="default",
              description="Default config for NaturalInstructions")
]
DEFAULT_CONFIG_NAME = "default"


def _info(self):
    return datasets.DatasetInfo(
        features=datasets.Features(
            {
                "Task": datasets.Value("string"),
                "Dataset": datasets.Value("string"),
                "subset": datasets.Value("string"),
                "Samples": [{
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "ground_truth": datasets.Value("string")
                }],
                "Instance": {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "ground_truth": datasets.Value("string")
                },
                "prompt": datasets.Value("string"),
                "query": datasets.Value("string"),
                "response": datasets.Value("string")
            }
        ),
        supervised_keys=None
    )


def split_generators(dl_manager):
    """Returns SplitGenerators."""
    if config.data_dir is None or config.task_configs is None:
        logger.error(
            "Please provide right input: data_dir or task_config_dir!")

    # split dir save datasets
    # task config to specify train,dev,test
    split_dir = config.data_dir
    task_configs = config.task_configs

    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "path": split_dir,
                "task_config": task_configs['train'],
                "max_num_instances_per_task":   config.max_num_instances_per_task,
                "subset": "train"
            }),
        datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={
                "path": split_dir,
                "task_config": task_configs['dev'],
                "max_num_instances_per_task":   config.max_num_instances_per_eval_task,
                "subset": "dev"
            }),
        datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "path": split_dir,
                "task_config": task_configs['test'],
                "max_num_instances_per_task": None,  # default load total test samples to test
                "subset": "test"
            }),
    ]


def load_dataset(dataset_path, labels_path):
    with open(dataset_path, encoding="utf-8") as task_f:
        s = task_f.read()
        instances = json.loads(s)
    with open(labels_path, encoding="utf-8") as labels_f:
        labels = json.load(labels_f)

    return instances, labels


def get_instruction(task):
    assert config.instruction_strategy in INSTRUCTION_STRATEGIES
    if config.num_examples is not None and config.num_examples > 0:
        task_instructions = config.instructions['few-shot'][task]
    else:
        task_instructions = config.instructions['zero-shot'][task]
    if config.instruction_strategy == "single":
        return task_instructions[0]
    else:
        return random.choice(task_instructions)


def sampling_dataset(instances, sampling_strategy, max_num_instances):
    if sampling_strategy == 'random' and max_num_instances is not None and max_num_instances >= 0:
        instances = instances[:max_num_instances]
    if max_num_instances != None and config.over_sampling and len(instances) < max_num_instances:
        origin_instances = instances.copy()
        while len(instances) < max_num_instances:
            instances.append(random.choice(origin_instances))
    if sampling_strategy == 'sample' and max_num_instances is not None and max_num_instances >= 0:
        origin_instances = instances.copy()
        num_samples = min(len(origin_instances), max_num_instances)
        instances = random.sample(origin_instances, num_samples)

    return instances


def load_NER_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    instances, labels = load_dataset(dataset_path, labels_path)

    sample_template = {
        "Task": "NER", "Dataset": dataset_name, "Samples": [], "subset": subset}

    labels_str = ', '.join([f"\"{label}\"" for label in labels])
    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()
        instruction = "def name_entity_recognition(input_text):\n    \"\"\" extract named entities from the input_text . \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Entity:\n        def __init__(self, entity_text: str, entity_type: str):\n            assert entity_text in input_text\n            assert entity_type in [{labels_str}]\n            self.entity_text = entity_text\n            self.entity_type = entity_type\n    entity_list = []\n    # extracted named entities\n"
        
        kv_pairs = []

        for entity in instance['entities']:
            if entity['type'] == 'NA' or entity['type'] == '':
                continue
            kv_pair = [entity['name'], entity['type']]
            kv_pairs.append(kv_pair)

        if len(kv_pairs) > 0:
            label = "\n".join([f"    entity_list.append(Entity(entity_text = \"{k}\", entity_type = \"{v}\"))"
                                     for (k, v) in kv_pairs])
        else:
            label = "    # None"

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": label,
            "instruction": instruction
        }

        yield example


def load_ES_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    # ES = Entity Span
    instances, labels = load_dataset(dataset_path, labels_path)

    sample_template = {"Task": "ES", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()
        instruction = "def entity_span(input_text):\n    \"\"\" extract span entities from the input_text . \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Entity_Span:\n        def __init__(self, entity_text: str):\n            assert entity_text in input_text\n            self.entity_text = entity_text\n    entity_list = []\n    # extracted span entities\n"
        entities = []

        for entity in instance['entities']:
            entities.append(entity["name"])

        if len(entities) > 0:
            label = "\n".join([f"    entity_list.append(Entity_Span(entity_text = \"{entity_name}\"))" for entity_name in entities])
        else:
            label = "    # None"

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": label,
            "instruction": instruction
        }

        if random.random() < AUX_PROB:
            yield example


def load_ET_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    # ET = Entity Type
    instances, labels = load_dataset(dataset_path, labels_path)

    sample_template = {"Task": "ET", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    labels_str = ', '.join([f"\"{label}\"" for label in labels])
    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()

        entities = []
        kv_pairs = []

        for entity in instance['entities']:
            if entity['type'] == 'NA' or entity['type'] == '':
                continue
            kv_pair = [entity['name'], entity['type']]
            kv_pairs.append(kv_pair)
            entities.append(entity["name"])

        entities_str = ", ".join([f"\"{entity_name}\"" for entity_name in entities])
        
        instruction = "def entity_typing(input_text):\n    \"\"\" type given named entities from the input_text . \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Entity:\n        def __init__(self, entity_text: str, entity_type: str):\n            assert entity_text in [{entities_str}]\n            assert entity_type in [{labels_str}]\n            self.entity_text = entity_text\n            self.entity_type = entity_type\n    entity_list = []\n    # typed named entities\n"

        if len(kv_pairs) > 0:
            label = "\n".join([f"    entity_list.append(Entity(entity_text = \"{k}\", entity_type = \"{v}\"))"
                                    for (k, v) in kv_pairs])
        else:
            label = "    # None"

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": label,
            "instruction": instruction
        }

        if random.random() < AUX_PROB:
            yield example


def load_EP_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    # EP = Entity Pair
    instances, labels = load_dataset(dataset_path, labels_path)
    sample_template = {"Task": "EP", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    labels_str = ', '.join([f"\"{label}\"" for label in labels])
    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()
        instruction = "def relation_triple(input_text):\n    \"\"\" extract the relations of span entities from the input text. \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Entity_Span:\n        def __init__(self, entity_text: str):\n            assert entity_text in input_text\n            self.entity_text = entity_text\n    class Relation_Triple:\n        def __init__(self, relation_type: str, head_entity: Entity_Span, tail_entity: Entity_Span):\n            assert relation_type in [{labels_str}]\n            self.relation_type = relation_type  \n            self.head_entity = head_entity\n            self.tail_entity = tail_entity\n    relation_list = []\n    # extracted triple relations\n"
        
        relation_pairs = []
        ground_truth_pairs = []

        for relation in instance['relations']:
            if relation['type'] == 'NA' or relation['type'] == '':
                continue
            relation_pair = [relation['type'], relation['head']['name'], relation['tail']['name']]
            ground_truth_pairs.append(relation_pair)
            relation_pairs.append(relation_pair)

        if len(relation_pairs) > 0:
            label = "\n".join([f"    relation_list.append(Relation_Triple(relation_type = \"{r}\", head_entity = Entity_Span(entity_text = \"{h}\"), tail_entity = Entity_Span(entity_text = \"{t}\")))"
                                    for (r, h, t) in relation_pairs])
        else:
            label = '    # None'

        if len(ground_truth_pairs) > 0:
            ground_truth = "\n".join([f"    relation_list.append(Relation_Triple(relation_type = \"{r}\", head_entity = Entity_Span(entity_text = \"{h}\"), tail_entity = Entity_Span(entity_text = \"{t}\")))"
                            for (r, h, t) in ground_truth_pairs])
        else:
            ground_truth = '    # None'

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": ground_truth,
            "instruction": instruction
        }

        if random.random() < AUX_PROB:
            yield example


def load_EPR_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    # EPR = Entity Pair Relationship
    instances, labels = load_dataset(dataset_path, labels_path)
    sample_template = {"Task": "EPR", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    labels_str = ', '.join([f"\"{label}\"" for label in labels])
    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()
        
        relation_pairs = []
        entity_pairs = []
        ground_truth_pairs = []

        for relation in instance['relations']:
            if relation['type'] == 'NA' or relation['type'] == '':
                ground_truth_pairs.append(
                    [relation['head']['name'], 'NA', relation['tail']['name']])
                continue
            relation_pair = [relation['head']['name'],
                             relation['type'], relation['tail']['name']]
            entity_pair = [relation['head']['name'], relation['tail']['name']]
            ground_truth_pairs.append(relation_pair)
            relation_pairs.append(relation_pair)
            entity_pairs.append(entity_pair)

        ep_name = ", ".join([f"(\"{h}\", \"{t}\")" for (h, t) in entity_pairs])
        
        instruction =  "def relation_classification(input_text):\n    \"\"\" extract the relations of given span entity pairs from the input text. \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Entity_Span:\n        def __init__(self, entity_text: str):\n            assert entity_text in input_text\n            self.entity_text = entity_text\n    class Relation_Triple:\n        def __init__(self, relation_type: str, head_entity: Entity_Span, tail_entity: Entity_Span):\n            assert relation_type in [{labels_str}]\n            assert (head_entity.entity_text, tail_entity.entity_text) in [{ep_name}]\n            self.relation_type = relation_type  \n            self.head_entity = head_entity\n            self.tail_entity = tail_entity\n    relation_list = []\n    # extracted triple relations\n"

        if len(relation_pairs) > 0:
            label = "\n".join([f"    relation_list.append(Relation_Triple(relation_type = \"{r}\", head_entity = Entity_Span(entity_text = \"{h}\"), tail_entity = Entity_Span(entity_text = \"{t}\")))"
                                    for (h, r, t) in relation_pairs])
        else:
            label = '    # None'

        if len(ground_truth_pairs) > 0:
            ground_truth = "\n".join([f"    relation_list.append(Relation_Triple(relation_type = \"{r}\", head_entity = Entity_Span(entity_text = \"{h}\"), tail_entity = Entity_Span(entity_text = \"{t}\")))"
                                    for (h, r, t) in ground_truth_pairs])
        else:
            logger.error("******Error item: {}******".format(instance))
            raise Exception(
                'Dataset Error:{}, No ground truth!'.format(dataset_name))

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": ground_truth,
            "instruction": instruction
        }

        if random.random() < AUX_PROB:
            yield example


def load_RE_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    instances, labels = load_dataset(dataset_path, labels_path)
    sample_template = {"Task": "RE", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    labels_str = ', '.join([f"\"{label}\"" for label in labels])
    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()
        instruction = "def relation_extraction(input_text):\n    \"\"\" extract the relations of named entities from the input text. \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Entity:\n        def __init__(self, entity_text: str, entity_type: str):\n            assert entity_text in input_text\n            assert entity_type in [\"NA\"]\n            self.entity_text = entity_text\n            self.entity_type = entity_type\n    class Relation:\n        def __init__(self, relation_type: str, head_entity: Entity, tail_entity: Entity):\n            assert relation_type in [{labels_str}]\n            self.relation_type = relation_type  \n            self.head_entity = head_entity\n            self.tail_entity = tail_entity\n    relation_list = []\n    # extracted relations\n"
        relation_pairs = []
        ground_truth_pairs = []

        for relation in instance['relations']:
            if relation['type'] == 'NA' or relation['type'] == '':
                ground_truth_pairs.append(
                    [relation['head']['name'], 'NA', relation['tail']['name']])
                continue
            relation_pair = [relation['head']['name'],
                             relation['type'], relation['tail']['name']]
            ground_truth_pairs.append(relation_pair)
            relation_pairs.append(relation_pair)

        if len(relation_pairs) > 0:
            label = "\n".join(f"    relation_list.append(Relation(relation_type = \"{r}\", head_entity = Entity(entity_text = \"{h}\", entity_type = \"NA\"), tail_entity = Entity(entity_text = \"{t}\", entity_type = \"NA\"))"
                                    for (h, r, t) in relation_pairs)
        else:
            label = '    # None'

        if len(ground_truth_pairs) > 0:
            ground_truth = "\n".join(f"    relation_list.append(Relation(relation_type = \"{r}\", head_entity = Entity(entity_text = \"{h}\", entity_type = \"NA\"), tail_entity = Entity(entity_text = \"{t}\", entity_type = \"NA\"))"
                                    for (h, r, t) in ground_truth_pairs)
        else:
            logger.error("******Error item: {}******".format(instance))
            raise Exception(
                'Dataset Error:{}, No ground truth!'.format(dataset_name))

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": ground_truth,
            "instruction": instruction
        }

        yield example


def load_EE_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    instances, labels = load_dataset(dataset_path, labels_path)
    sample_template = {"Task": "EE", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    labels_str = {
        "Event type": 
            ", ".join([f"\"{label}\"" for label in labels[0].split(", ")]), 
        "Arguments type": 
            ", ".join([f"\"{label}\"" for label in labels[1].split(", ")])
        }
    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()
        instruction = "def event_extraction(input_text):\n    \"\"\" extract the events from the input text. \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Event_Trigger:\n        def __init__(self, event_type: str, trigger: str):\n            assert event_type in [{labels_str['Event type']}]\n            assert trigger in input_text\n            self.event_type = event_type\n            self.trigger = trigger\n    class Event_Argument:\n        def __init__(self, argument_name: str, argument_role: str):\n            assert argument_name in input_text\n            assert argument_role in [{labels_str['Arguments type']}]\n            self.argument_name = argument_name  \n            self.argument_role = argument_role\n    class Event:\n        def __init__(self, event_trigger: Event_Trigger, event_arguments: list):\n            self.event_trigger = event_trigger\n            self.event_arguments = event_arguments        \n    event_list = []\n    # extracted events\n"
        
        
        event_pairs = []

        for k, event in enumerate(instance['events']):
            instance['events'][k]['trigger'] = event['trigger'].replace(
                "'", SINGLE_QUOTES_SUBSTITUTE)
            instance['events'][k]['type'] = event['type'].replace(
                "'", SINGLE_QUOTES_SUBSTITUTE)

            if event['type'] == 'NA' or event['type'] == '':
                continue
            event_type = event['type']
            event_trigger = event['trigger']
            event_arguments = [f"Event_Argument(argument_name = \"{argument['name']}\", argument_role = \"{argument['role']}\")" for
                               argument in event['arguments']]

            event_arguments = "" if not event_arguments else ", ".join(
                event_arguments)
            event_pair = [event_type, event_trigger, event_arguments]
            event_pairs.append(event_pair)

        if len(event_pairs) > 0:
            label = "\n".join([f"    event_list.append(Event(event_trigger = Event_Trigger(event_type = \"{type}\", trigger = \"{trigger}\"), event_arguments = [{arguments}]))"
                              for (type, trigger, arguments) in event_pairs])
        else:
            label = '    # None'

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": label,
            "instruction": instruction
        }

        yield example


def load_EET_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    instances, labels = load_dataset(dataset_path, labels_path)
    sample_template = {"Task": "EET", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    labels_str = ", ".join([f"\"{label}\"" for label in labels.keys()])
    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        example = sample_template.copy()
        instruction = "def event_detection(input_text):\n    \"\"\" extract the event type and its trigger word from the input text. \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Event_Trigger:\n        def __init__(self, event_type: str, trigger: str):\n            assert event_type in [{labels_str}]\n            assert trigger in input_text\n            self.event_type = event_type\n            self.trigger = trigger\n    event_trigger_list = []\n    # extracted event triggers\n"
        
        event_pairs = []

        for k, event in enumerate(instance['events']):
            instance['events'][k]['trigger'] = event['trigger'].replace(
                "'", SINGLE_QUOTES_SUBSTITUTE)
            instance['events'][k]['type'] = event['type'].replace(
                "'", SINGLE_QUOTES_SUBSTITUTE)

            if event['type'] == 'NA' or event['type'] == '':
                continue
            event_type = event['type']
            event_trigger = event['trigger']
            event_pair = [event_type, event_trigger]
            event_pairs.append(event_pair)

        if len(event_pairs) > 0:
            label = "\n".join([f"    event_trigger_list.append(Event_Trigger(event_type = \"{type}\", trigger = \"{trigger}\"))"
                                    for (type, trigger) in event_pairs])
        else:
            label = '    # None'

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": label,
            "instruction": instruction
        }

        yield example


def load_EEA_dataset(dataset_path, labels_path, dataset_name, sampling_strategy, max_num_instances, subset):
    instances, labels = load_dataset(dataset_path, labels_path)
    sample_template = {"Task": "EEA", "Dataset": dataset_name,
                       "Samples": [], "subset": subset}

    instances = sampling_dataset(
        instances, sampling_strategy, max_num_instances)

    for idx, instance in enumerate(instances):
        if len(instance['events']) > 1:
            raise "Error: EEA dataset should only have one event."
        labels_str = ', '.join([f"\"{label}\"" for label in labels[instance['events'][0]['type']]])
        example = sample_template.copy()
        event = instance['events'][0]

        instruction = "def event_arguments_extraction(input_text):\n    \"\"\" extract the event arguments with given event type and trigger from the input text. \"\"\"\n"
        input = f"    input_text = \"{instance['sentence']}\"\n    class Event_Trigger:\n        def __init__(self, event_type: str, trigger: str):\n            assert trigger in input_text\n            self.event_type = event_type\n            self.trigger = trigger\n    class Event_Argument:\n        def __init__(self, argument_name: str, argument_role: str):\n            self.event = Event_Trigger(event_type = \"{event['type']}\", trigger = \"{event['trigger']}\")\n            assert argument_name in input_text\n            assert argument_role in [{labels_str}]\n            self.argument_name = argument_name  \n            self.argument_role = argument_role\n    event_argument_list = []\n    # extracted event arguments\n"
        
        event_arguments = [f"    event_argument_list.append(Event_Argument(argument_name = \"{argument['name']}\", argument_role = \"{argument['role']}\"))".format(argument['name'], argument['role']) for
                           argument in event['arguments']]
        label = "    # None" if not event_arguments else "\n".join(event_arguments)

        example["Instance"] = {
            "id": str(idx),
            "sentence": input,
            "label": label,
            "ground_truth": label,
            "instruction": instruction
        }
        yield example


def generate_examples(path=None, task_config=None, max_num_instances_per_task=None, subset=None, output=None, zero_shot=False):
    """Yields examples."""
    print(f"Generating tasks from = {path}")
    instances = []
    sft_data = []
    max_length = 0
    
    for task in task_config:
        print("\n", "=" * 15, f"task: {task}", "=" * 15)
        if task == "NER":
            load_func = load_NER_dataset
        elif task == 'RE':
            load_func = load_RE_dataset
        elif task == 'EE':
            load_func = load_EE_dataset
        elif task == 'ES':
            load_func = load_ES_dataset
        elif task == 'ET':
            load_func = load_ET_dataset
        elif task == 'EP':
            load_func = load_EP_dataset
        elif task == 'EPR':
            load_func = load_EPR_dataset
        elif task == 'EET':
            load_func = load_EET_dataset
        elif task == 'EEA':
            load_func = load_EEA_dataset
        else:
            raise ValueError(
                "Unsupport {} task, plz check {} task config!".format(task, subset))
            
        # load dataset
        for dataset in task_config[task]:
            ds_name = dataset["dataset name"]
            sampling_strategy = dataset.get("sampling strategy", "random")
            ds_path = os.path.join(path, task, ds_name, subset + '.json')
            labels_path = os.path.join(path, task, ds_name, 'labels.json')
            print(ds_path)
            assert os.path.exists(ds_path)
            assert os.path.exists(labels_path)

            idx = -1
            
            for sample in load_func(ds_path, labels_path, ds_name, sampling_strategy, max_num_instances_per_task,
                                    subset):
                idx += 1
                instances.append(sample)
                sft_item, length = seq2seq_call(sample)
                sft_data.append(sft_item)
                if length > max_length:
                    max_length = length
    
    os.makedirs(f'processed/{output}', exist_ok=True)
    with open(f'processed/{output}/{subset}.json', encoding='utf-8', mode='w') as f:
        f.write(json.dumps(instances))  
    os.makedirs(f'data/{output}_{subset}', exist_ok=True)
    with open(f'data/{output}_{subset}/examples.json', encoding='utf-8', mode='w') as f:
        f.write(json.dumps(sft_data))  
    os.makedirs(f'data', exist_ok=True)
    with open('data/dataset_info.json', 'r') as file:
        dataset_info = json.load(file) 
    if f'{output}_{subset}' not in dataset_info:
        dataset_info[f'{output}_{subset}'] = {
            "script_url": f'{output}_{subset}',
            "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history"
            }  
        } 
    with open('data/dataset_info.json', encoding='utf-8', mode='w') as f:
        f.write(json.dumps(dataset_info))  
        
    print(f'data_dir: processed/{output}')
    print(f'sft_dir: data/{output}_{subset}/examples.json')
    print(f"data size: {len(sft_data)}")
    print(f"max length: {max_length}")  
                
def _get_instruction(instance):
    # "instructions \n options \n {0} \n Answer: "
    instruction = instance['Instance']["instruction"]
    content = instance['Instance']['sentence']

    # TODO, support few shot
    # add few shot samples
    samples = ''
    if len(instance['Samples']) > 0:
        raise Exception('Few shot is coming soon...')
    if samples:
        content = samples + content
    # TODO, fix bug
    try:
        instruction = instruction.format(content)
    finally:
        return instruction             
                
def seq2seq_call(instance):

    label = instance['Instance']['label']
    instruction = _get_instruction(instance)

    return {"instruction": "", "input": instruction, "output": label, "history": []}, len((instruction + label).split())


if __name__ == '__main__':
    print(os.listdir("IE_INSTRUCTIONS"))
    config = UIEConfig(
        data_dir="IE_INSTRUCTIONS",
        instruction_file="configs/instruction_config_code.json",
        instruction_strategy="single",
        task_config_dir="configs/new_multi_task_configs"
        )
    print(config.data_dir)
    OUTPUT_PATH = 'InstructionUIE'
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    generate_examples(path="IE_INSTRUCTIONS", task_config=config.task_configs['train'], 
                      max_num_instances_per_task=1000, subset="train", output='code_sft_new', zero_shot=False)
    generate_examples(path="IE_INSTRUCTIONS", task_config=config.task_configs['test'],
                      max_num_instances_per_task=1000, subset="test", output='code_sft_new', zero_shot=False)
    # generate_examples(path="IE_INSTRUCTIONS", task_config=config.task_config,
    #                   max_num_instances_per_task=200, subset="test", output='zero_shot.json', zero_shot=True)
    print("exit")
