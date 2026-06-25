import os, json
from PIL import Image
from PIL.Image import Image as PILImage
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from typing import NamedTuple
from qwen_vl_utils import process_vision_info


diagram_schema_by_dataset = {
    'fc_a': {
        'diagram_type': 'flowchart',
        'node_classes': ['connection', 'data', 'decision', 'process', 'terminator'],
        'relation_classes': 'null',
        'group_classes': 'null'
    },
    'hdBPMN-icdar2021': {
        'diagram_type': 'bpmn',
        'node_classes': ['dataObject', 'dataStore', 'event', 'eventBasedGateway', 'exclusiveGateway', 'messageEvent', 'parallelGateway', 'subProcess', 'task', 'timerEvent'],
        'relation_classes': ['dataAssociation', 'messageFlow', 'sequenceFlow'],
        'group_classes': ['lane', 'pool']
    },
    'cbd': {
        'diagram_type': 'flowchart',
        'node_classes': 'null',
        'relation_classes': 'null',
        'group_classes': 'null'
    },
    'didi': {
        'diagram_type': 'flowchart',
        'node_classes': ['box', 'diamond', 'octagon', 'oval', 'parallelogram'],
        'relation_classes': 'null',
        'group_classes': 'null'
    },
    'fa': {
        'diagram_type': 'finite automata',
        'node_classes': ['final state', 'state'],
        'relation_classes': 'null',
        'group_classes': 'null'
    },
    'sems': {
        'diagram_type': 'structural equation modeling',
        'node_classes': ['Construct', 'Item', 'Error'],
        'relation_classes': ['unidirectional', 'bidirectional'],
        'group_classes': 'null'
    },
    'sap-sam-uml': {
        'diagram_type': 'UML',
        'node_classes': ['ComplexClass', 'SimpleClass', 'Note', 'Interface', 'Enumeration'],
        'relation_classes': ['Generalization', 'Composition', 'Aggregation', 'UndirectedAssociation', 'Annotation Edge', 'Realization', 'Dependency', 'DirectedAssociation'],
        'group_classes': 'null'
    },
    'sap-sam-bpmn': {
        'diagram_type': 'bpmn',
        'node_classes': ['Subprocess', 'StartEvent', 'Task', 'EndEvent', 'IntermediateTimerEvent', 'IntermediateMessageEventCatching', 'EndTerminateEvent', 'Exclusive_Databased_Gateway', 'AND_Gateway', 'IntermediateMessageEventThrowing', 'StartMessageEvent', 'Exclusive_Eventbased_Gateway', 'TextAnnotation', 'DataObject', 'IntermediateLinkEventThrowing', 'IntermediateErrorEvent', 'CollapsedSubprocess', 'EndMessageEvent', 'IntermediateConditionalEvent', 'OR_Gateway', 'IntermediateEvent', 'StartTimerEvent'],
        'relation_classes': ['SequenceFlow', 'MessageFlow', 'Association_Unidirectional', 'Association_Undirected'],
        'group_classes': ['Pool', 'Lane', 'CollapsedPool']
    },
    'sap-sam-petrinet': {
        'diagram_type': 'petri net',
        'node_classes': ['Place', 'Transition', 'VerticalEmptyTransition'],
        'relation_classes': ['Arc'],
        'group_classes': 'null'
    },
    'synth_fc': {
        'diagram_type': 'flowchart',
        'node_classes': ['oval', 'circle', 'parallelogram', 'box', 'document', 'ellipse', 'hexagon', 'diamond'],
        'relation_classes': 'null',
        'group_classes': 'null'
    },
    'synth_dir_graphs': {
        'diagram_type': 'directed graph',
        'node_classes': ['node'],
        'relation_classes': ['unidirectional', 'bidirectional'],
        'group_classes': 'null'
    }
}


all_datasets = [{'images': ['ex00_writer0075.jpg',
   'ex06_writer0083.jpg',
   'ex07_writer0078.jpg',
   'ex07_writer0076.jpg',
   'ex02_writer0067.jpg',
   'ex06_writer0066.jpg',
   'ex06_writer0071.jpg',
   'ex04_writer0075.jpg',
   'ex00_writer0070.jpg',
   'ex02_writer0079.jpg',
   'ex01_writer0084.jpg',
   'ex00_writer0073.jpg',
   'ex06_writer0067.jpg',
   'ex06_writer0075.jpg',
   'ex04_writer0081.jpg',
   'ex02_writer0075.jpg',
   'ex07_writer0079.jpg',
   'ex06_writer0072.jpg',
   'ex07_writer0067.jpg',
   'ex00_writer0065.jpg'],
  'dataset_name': 'hdBPMN-icdar2021',
  'split': 'val',
  'label_dataset_name': 'hdBPMN-icdar2021'},
 {'images': ['writer33_8.png',
   'writer9_2_.png',
   'writer31_3.png',
   'writer5_11.png',
   'writer26_14.png',
   'writer17_14.png',
   'writer2_2.png',
   'writer16_9.png',
   'writer2_1.png',
   'writer1_7.png',
   'writer26_10.png',
   'writer33_9.png',
   'writer24_12.png',
   'writer7_8.png',
   'writer29_9.png',
   'writer18_10.png',
   'writer11_2_.png',
   'writer34_6.png',
   'writer6_8.png',
   'writer7_7.png'],
  'dataset_name': 'fc_a',
  'split': 'train',
  'label_dataset_name': 'fc_a'},
 {'images': ['Normal (191).png',
   'Normal (209).png',
   'Normal (190).png',
   'Connect (68).png',
   'Break (68).png',
   'Normal (218).png',
   'Normal (182).png',
   'Connect (76).png',
   'Normal (219).png',
   'Normal (216).png',
   'Normal (205).png',
   'Normal (234).png',
   'Connect (83).png',
   'Break (63).png',
   'Normal (221).png',
   'Normal (223).png',
   'Normal (194).png',
   'Break (75).png',
   'Break (64).png',
   'Normal (239).png'],
  'dataset_name': 'cbd',
  'split': 'val',
  'label_dataset_name': 'cbd'},
 {'images': ['512b744a19c3dfc4ed55fded68e9f3a393c3da28.png',
   '9ba42de10df5b8244d385424c1a63e58d3656382.png',
   'e1651080ede0d323e87cc034e314208ec3bd8cd9.png',
   'c2468ec6700b65650333425190d544f337eaaa9c.png',
   '2c57100256d5109effcffa12a38c5eb3e53d946a.png',
   'c84eb0e50aa92f2bb988cf6e8efa332a868879db.png',
   '63af7dc992143959eced903dd7519d4f8371ef84.png',
   '825db7987c74a8370d4125c43be0b243bb2776d2.png',
   'd35d2a8ea5f4b81c67aaa540b0fe061f99980830.png',
   '9cf1fdfc2b31367443473f785610385ca4131b69.png',
   'b08bc40d4018d3870797a7ae92256ce8a6867460.png',
   '403876e7b1be238c860297682fd16b805ef8ba6d.png',
   '37c84301ba36955e8f3125ed51f58c3fe401e4d1.png',
   '2269e514cfda83117886e6008febc4886cb021e1.png',
   '596ef17659035fb13a3ddb041b44918ce79c75dd.png',
   '876a157f75d8b921c55588ea3f0a68cb3a1b3b3f.png',
   '57f6667c601780de9705093a8c7479051a9557b4.png',
   '007716092aa5e13d2b4d345c5c32bb4b571fb962.png',
   '35cdb1566def86a5d3b44c2103a1bc3782a843e0.png',
   'df340b7a3d7be209c08d78f68f04ea2514c42298.png'],
  'dataset_name': 'didi',
  'split': 'val',
  'label_dataset_name': 'didi'},
 {'images': ['writer006_fa_007.png',
   'writer010_fa_001.png',
   'writer010_fa_003.png',
   'writer004_fa_007.png',
   'writer005_fa_003.png',
   'writer005_fa_012.png',
   'writer004_fa_006.png',
   'writer001_fa_007.png',
   'writer006_fa_008.png',
   'writer000_fa_006.png',
   'writer000_fa_007.png',
   'writer003_fa_006.png',
   'writer000_fa_009.png',
   'writer005_fa_005.png',
   'writer002_fa_011.png',
   'writer004_fa_011.png',
   'writer006_fa_011.png',
   'writer009_fa_003.png',
   'writer007_fa_005.png',
   'writer009_fa_011.png'],
  'dataset_name': 'fa',
  'split': 'train',
  'label_dataset_name': 'fa'}]

few_shot_examples = {
    'fa': ['train/writer002_fa_002.png', 'train/writer008_fa_010.png', 'train/writer001_fa_003.png', 'train/writer010_fa_004.png', 'train/writer009_fa_008.png'],
    'hdBPMN-icdar2021': ['train/ex09_writer0028.jpg', 'train/ex07_writer0013.jpg', 'train/ex00_writer0044.jpg', 'train/ex07_writer0058.jpg', 'train/ex05_writer0019.jpg'],
    'cbd': ['train/Break (21).png', 'train/Connect (47).png', 'train/Break (55).png', 'train/Normal (176).png', 'train/Connect (64).png'],
    'didi': ['0a0ad20ab28f166d189e704e0cbcaafa381c985f.png', '00a1c9e2f7f791f891de646a41f91427d18e18df.png'],
    'fc_a': ['train/writer1_1.png', 'train/writer34_5.png', 'train/writer6_12.png', 'train/writer13_14.png', 'train/writer26_3.png'],
    'sems': ['train/sem_figure_27.jpg', 'train/sem_figure_819.jpg', 'train/isre.1060.0094_figure_17_8.png', 'train/Information-Technology-and-organizational-innovation-_2020_The-Journal-of-St_figure_15_14.png', 'train/ijerph-17-02170_figure_7_4.png'],
    'sap-sam-bpmn': ['train/1bbed2ff7c72407a9ee214ee15ef00c8.png', 'train/1bb8521fb80e4b4ba5e23915176da38e.png', 'train/1bc6a5a5412b468783fa2bdccc68b8e2.png', 'train/1bf0b4ea095e48dba35b44540456dc00.png', 'train/1c3de2bad7af4d738fc8403cd9969aa9.png'],
    'sap-sam-petrinet': ['train/1d32ba58648a4c038b2ca716e4bd2f77.png', 'train/643b4740a0d1433e9f56d6305678587d.png'],
    'sap-sam-uml': ['train/1c472f2deccf44bcb705c63780d77137.png', 'train/64e15060ad044e76b2ec64a241156409.png', 'train/1bb3e5fe23f64712aaf592841e8fd7b6.png', 'train/1bc224c87ff8466c90ecba1509f66cf5.png', 'train/e121b751eb0a4c77a62260c2263e5117.png'],
    'synth_fc': ['train/flowchart_1135_13.png', 'train/flowchart_0011_20.png', 'train/flowchart_1731_10.png', 'train/flowchart_2993_7.png', 'train/flowchart_0970_8.png'],
    'synth_dir_graphs': ['train/sem_11538_11_polyline_0_dot.png', 'train/sem_11834_15_polyline_0_dot.png', 'train/sem_4093_15_polyline_0_dot.png', 'train/sem_8864_13_polyline_0_dot.png', 'train/sem_11670_15_spline_0_dot.png']
}

few_shot_labels = {
    'fa': ['train/writer002_fa_002.json', 'train/writer008_fa_010.json', 'train/writer001_fa_003.json', 'train/writer010_fa_004.json', 'train/writer009_fa_008.json'],
    'hdBPMN-icdar2021': ['train/ex09_writer0028.json', 'train/ex07_writer0013.json', 'train/ex00_writer0044.json', 'train/ex07_writer0058.json', 'train/ex05_writer0019.json'],
    'cbd': ['train/Break (21).json', 'train/Connect (47).json', 'train/Break (55).json', 'train/Normal (176).json', 'train/Connect (64).json'],
    'didi': ['train/0a0ad20ab28f166d189e704e0cbcaafa381c985f.json', 'train/00a1c9e2f7f791f891de646a41f91427d18e18df.json'],
    'fc_a': ['train/writer1_1.json', 'train/writer34_5.json', 'train/writer6_12.json', 'train/writer13_14.json', 'train/writer26_3.json'],
    'sems': ['train/sem_figure_27.json', 'train/sem_figure_819.json', 'train/isre.1060.0094_figure_17_8.json', 'train/Information-Technology-and-organizational-innovation-_2020_The-Journal-of-St_figure_15_14.json', 'train/ijerph-17-02170_figure_7_4.json'],
    'sap-sam-bpmn': ['train/1bbed2ff7c72407a9ee214ee15ef00c8.json', 'train/1bb8521fb80e4b4ba5e23915176da38e.json', 'train/1bc6a5a5412b468783fa2bdccc68b8e2.json', 'train/1bf0b4ea095e48dba35b44540456dc00.json', 'train/1c3de2bad7af4d738fc8403cd9969aa9.json'],
    'sap-sam-petrinet': ['train/1d32ba58648a4c038b2ca716e4bd2f77.json', 'train/643b4740a0d1433e9f56d6305678587d.json'],
    'sap-sam-uml': ['train/1c472f2deccf44bcb705c63780d77137.json', 'train/64e15060ad044e76b2ec64a241156409.json', 'train/1bb3e5fe23f64712aaf592841e8fd7b6.json', 'train/1bc224c87ff8466c90ecba1509f66cf5.json', 'train/e121b751eb0a4c77a62260c2263e5117.json'],
    'synth_fc': ['train/flowchart_1135_13.json', 'train/flowchart_0011_20.json', 'train/flowchart_1731_10.json', 'train/flowchart_2993_7.json', 'train/flowchart_0970_8.json'],
    'synth_dir_graphs': ['train/sem_11538_11_polyline_0_dot.json', 'train/sem_11834_15_polyline_0_dot.json', 'train/sem_4093_15_polyline_0_dot.json', 'train/sem_8864_13_polyline_0_dot.json', 'train/sem_11670_15_spline_0_dot.json']
}


OUTDIR = "/content/drive/MyDrive/vllm - experiment/outputs_qwen"
os.makedirs(OUTDIR, exist_ok=True)

N_RUNS          = 5
BASE_DIR        = './datasets'
BASE_DIR_LABELS = './labels'
MODEL_NAME      = "Qwen/Qwen3.5-27B"  
MAX_TOKENS      = 8192

# Resolution budget per image.
# Qwen-VL tiles images dynamically between min_pixels and max_pixels.
MIN_PIXELS            = 256 * 28 * 28
MAX_PIXELS            = 2048 * 28 * 28

MAX_IMAGES_PER_PROMPT = 6


class ModelRequestData(NamedTuple):
    prompt: str
    image_data: list[PILImage]


def load_qwen3vl(image_path, strat, dataset_name, processor):

    prompts = {}
    for fname in os.listdir('./prompts'):
        if fname.endswith('.txt'):
            with open(os.path.join('./prompts', fname), 'r') as f:
                prompts[fname[:-4]] = f.read().strip()

    prompt_key = strat + '-gt_names_uml' if 'uml' in dataset_name else strat + '-gt_names'
    question = prompts[prompt_key]

    question = (question
        .replace("{allowed_node_classes}",     str(diagram_schema_by_dataset[dataset_name]['node_classes']))
        .replace("{allowed_relation_classes}", str(diagram_schema_by_dataset[dataset_name]['relation_classes']))
        .replace("{allowed_group_classes}",    str(diagram_schema_by_dataset[dataset_name]['group_classes']))
        .replace("{diagram_type}",             str(diagram_schema_by_dataset[dataset_name]['diagram_type'])))

    example_labels = few_shot_labels[dataset_name]
    example_images = few_shot_examples[dataset_name]

    path2images = []

    if strat == 'one_shot':
        path2images.append(f'{BASE_DIR}/{dataset_name}/{example_images[0]}')
        with open(f'./labels/{dataset_name}/{example_labels[0]}') as f:
            label_data = json.load(f)
            question = question.replace('<ground_truth>', json.dumps(label_data, indent=2))

    elif strat == 'two_shot':
        for i in range(2):
            path2images.append(f'{BASE_DIR}/{dataset_name}/{example_images[i]}')
            with open(f'./labels/{dataset_name}/{example_labels[i]}') as f:
                label_data = json.load(f)
                question = question.replace(f'<ground_truth{i+1}>', json.dumps(label_data, indent=2))

    elif strat == 'five_shot':
        for i in range(5):
            path2images.append(f'{BASE_DIR}/{dataset_name}/{example_images[i]}')
            with open(f'./labels/{dataset_name}/{example_labels[i]}') as f:
                label_data = json.load(f)
                question = question.replace(f'<ground_truth{i+1}>', json.dumps(label_data, indent=2))


    shot_count = {'zero_shot': 0, 'one_shot': 1, 'two_shot': 2, 'five_shot': 5}

    path2images.append(image_path)

    pil_images = [Image.open(p).convert("RGB") for p in path2images]

    image_content = [
        {
            "type": "image",
            "image": img,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        }
        for img in pil_images
    ]

    messages = [
        {
            "role": "user",
            "content": [
                *image_content,
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)

    return ModelRequestData(
        prompt=prompt,
        image_data=image_inputs,
    )


def get_batch(strat, dataset_name, processor):
    imgfiles = sorted(os.listdir(os.path.join(BASE_DIR, dataset_name, 'test')))
    prompt_batch = []
    for img_name in imgfiles:
        req = load_qwen3vl(
            os.path.join(BASE_DIR, dataset_name, 'test', img_name),
            strat=strat,
            dataset_name=dataset_name,
            processor=processor,
        )
        prompt_batch.append({
            "prompt": req.prompt,
            "multi_modal_data": {"image": req.image_data},
        })
    return prompt_batch, imgfiles


def run_generate(llm):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    for t in [0.1]:
        if t is None:
            sampling_params = SamplingParams(
                max_tokens=MAX_TOKENS,
            )
        else:
            sampling_params = SamplingParams(
                temperature=t,
                max_tokens=MAX_TOKENS,
            )

        for strat in ['five_shot', 'zero_shot', 'two_shot']:
            for d in ['synth_fc', 'synth_dir_graphs']:

                for i in range(N_RUNS):


                    temp_dir = "temp=0.1" if t==0.1 else "temp=default"
                    out_dir = os.path.join(OUTDIR, d, strat, temp_dir)
                    os.makedirs(out_dir, exist_ok=True)

                    extraction_prompts, image_names = get_batch(strat, d, processor)
                    print(f"{strat}: {d} | temp={t} | run {i+1}/{N_RUNS} "
                          f"| {len(image_names)} images")

                    outputs = llm.generate(extraction_prompts, sampling_params)

                    for fname, output in zip(image_names, outputs):
                        stem = os.path.splitext(fname)[0]
                        out_path = os.path.join(out_dir, f"{stem}_temp={t}_run{i+1}.txt")
                        with open(out_path, 'w') as f:
                            f.write(output.outputs[0].text)


llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    max_model_len=32768,
    max_num_seqs=300,
    limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT},
    gpu_memory_utilization=0.9,
)

run_generate(llm)