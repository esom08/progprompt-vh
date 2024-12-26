# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""
This script evaluates plan generation using openAI LLMs
for the VirtualHome environment tasks
virtualhome simul 상태에서 plan 생성
"""

import sys
sys.path.append("virtualhome/simulation")
sys.path.append("virtualhome/demo")
sys.path.append("virtualhome")

import argparse
import os
import os.path as osp
import random

from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from virtualhome.demo.utils_demo import *

import openai
import json
import time

from utils_execute import *


#얘는 결과가 어케 나왔는지 오차율 느낌으로 구하는 함수
def eval(final_states, 
         final_states_GT, 
         initial_states, 
         test_tasks,
         exec_per_task,
         log_file):
    
    ## the evaluation criteria is not perfect
    ## since sometimes the tasks are underspecified, like which object to interact with
    ## for example "turn off lightswitch" could happen in multiple locations
    ## the evaluation happens w.r.t one possible valid state
    ## that the annotator provides
    ## GT는 ground truth인듯
    sr = [] 
    unsatif_conds = []; unchanged_conds = []
    total_goal_conds = []; total_unchanged_conds = []
    results = {}
    #zip : 동일 인덱스 튜플로 묶은 iterator 반환
    for g, g_gt, g_in, d in zip(final_states, final_states_GT, initial_states, test_tasks):
        '''
        g:final_states
        g_gt:final_states_GT
        g_in:initial_states
        d:test_tasks        
        '''
        #initial_state 정리
        obj_ids = dict([(node['id'], node['class_name']) for node in g_in['nodes']]) 
        # initial_state['nodes']을 순회하면서 그거의 id와 class_name을 튜플 후 dict로 저장함
        # (initial_state) id : class name
        relations_in = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in g_in["edges"]])
        # (initial_state) 'from_id relation_type to_id'형태로 중복된 거 제거된 상태로 저장함(집합)
        obj_states_in = set([node['class_name'] + ' ' + st for node in g_in['nodes'] for st in node['states']])
        # (initial_state)의 'nodes'에 있는 'state'를 돌면서 집합으로 'node['class_name'] node['states']'를 저장함
        
        #final _state 정리
        obj_ids = dict([(node['id'], node['class_name']) for node in g['nodes']])
        relations = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in g["edges"]])
        obj_states = set([node['class_name'] + ' ' + st for node in g['nodes'] for st in node['states']])

        #final_state_GT 정리
        obj_ids = dict([(node['id'], node['class_name']) for node in g_gt['nodes']])
        relations_gt = set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in g_gt["edges"]])
        obj_states_gt = set([node['class_name'] + ' ' + st for node in g_gt['nodes'] for st in node['states']])



        log_file.write(f"\nunsatisfied state conditions: relations: {(relations_gt - relations_in) - (relations - relations_in)}, object states: {(obj_states_gt - obj_states_in) - (obj_states - obj_states_in)}")
        '''
            unsatisfied state conditions: 
                relations: {(relations_gt - relations_in) - (relations - relations_in)},
                object states: {(obj_states_gt - obj_states_in) - (obj_states - obj_states_in)}

            이런 형태로 저장하겠다는건데  finalstate 나 finalstate_GT의 관계나 상태가 초기값에 없으면 안되니깐
            그 없는 값들을 log_file 에 저장하겠다는거
        '''
        unsatif_conds.append((len((relations_gt - relations_in) - (relations - relations_in))+len((obj_states_gt - obj_states_in) - (obj_states - obj_states_in))))
        ## 로그파일에 적은대로 만족 못시키는 애들 개수 추가하기
        ## (기준goal-initial)-(결과-initial) 느낌인뎅 겹치는걸 제외한건가?
        total_goal_conds.append(len(relations_gt - relations_in)+len(obj_states_gt - obj_states_in))
        ## 기준goal - 처음상태
        sr.append(1-unsatif_conds[-1]/total_goal_conds[-1])
        ## satisfaction rate
        ## 가장 최근값 기준으로 1-못만족/목표 즉 만족시키는 비율을 찾겠다는 뜻
        unchanged_conds.append((len(relations_gt.intersection(relations_in) - relations)+len(obj_states_gt.intersection(obj_states_in) - obj_states)))
        ## 교집합 찾아서 안변한 것들 넣기 final_state가 교집합에 있으면 제외해주기
        total_unchanged_conds.append(len(relations_gt.intersection(relations_in))+len(obj_states_gt.intersection(obj_states_in)))
        ## 얜 gt와 initial의 교집합만 봄
        
        #결과값 출력
        results[d] = {'PSR': sr[-1],
                        "SR": sr[-1:].count(1.0),
                        "Precision": 1-unchanged_conds[-1]/total_unchanged_conds[-1],
                        "Exec": exec_per_task[-1]
                        }

    
    results["overall"] = {'PSR': sum(sr)/len(sr),
                            "SR": sr.count(1.0)/len(sr),
                            "Precision": 1-sum(unchanged_conds)/sum(total_unchanged_conds),
                            "Exec": sum(exec_per_task)/len(exec_per_task)
                            }
    return results


def planner_executer(args):

    # initialize env
    comm = UnityCommunication(file_name=args.unity_filename, 
                              port=args.port, 
                              x_display=args.display)
    
    # prompt example environment is set to env_id 0
    comm.reset(0)

    _, env_graph = comm.environment_graph()
    ## response['success'], json.loads(response['message'])
    ## return: pair success (bool), graph: (dictionary)
    ## current state에 대해 environment graph를 반환
    '''
    이런식으로 되어있음
    {
   "nodes":[
      {
         "id":1,
         "class_name":"character",
         "states":[ 얘는 open/closed/on/onff

         ],
         "properties":[ 얘는 뭐 obj가 행동을 할 수 있는지

         ],
         "category":"Person"
      },
      {
         "id":2,
         "class_name":"kitchen",
         "states":[

         ],
         "properties":[

         ],
         "category":"Room"
      }
   ],
   "edges":[
      {
         "from_id":1,
         "to_id":2,
         "relation_type":"INSIDE"
      }
   ]
   '''
    obj = list(set([node['class_name'] for node in env_graph["nodes"]]))
    ## 그래프의 node의 class_name을 중복 제거하고 list로 저장
    ## 현재 환경에 있는 물건들임

    # define available actions and append available objects from the env
    ## 가능한 actions과 사용할 수 있는 obj 전달할거임
    prompt = f"from actions import turnright, turnleft, walkforward, walktowards <obj>, walk <obj>, 
                run <obj>, grab <obj>, switchon <obj>, switchoff <obj>, open <obj>, close <obj>, lookat <obj>, 
                sit <obj>, standup, find <obj>, turnto <obj>, drink <obj>, pointat <obj>, watch <obj>, 
                putin <obj> <obj>, putback <obj> <obj>"
    prompt += f"\n\nobjects = {obj}"

    # load train split for task examples
    ## 뭐 wash_clothes면 미리 저장되어 있는 이거의 plan을 prompt_egs에 dict 형태로 넣음
    with open(f"{args.progprompt_path}/data/pythonic_plans/train_complete_plan_set.json", 'r') as f:
        tmp = json.load(f)
        prompt_egs = {}
        for k, v in tmp.items():
            prompt_egs[k] = v
    # print("Loaded %d task example" % len(prompt_egs.keys()))

    ## define the prompt example task setting ##

    # default examples from the paper
    if args.prompt_task_examples == "default":
        default_examples = ["put_the_wine_glass_in_the_kitchen_cabinet",
                            "throw_away_the_lime",
                            "wash_mug",
                            "refrigerate_the_salmon",
                            "bring_me_some_fruit",
                            "wash_clothes",
                            "put_apple_in_fridge"]
        for i in range(args.prompt_num_examples):
            prompt += "\n\n" + prompt_egs[default_examples[i]]
            ## 명령할 task를 default로 설정한다면 default examples의 미리 정해진 plan을 prompt 로 전달함

    # random egs - change seeds
    ## random으로 지정하면 random으로 task 넘김
    if args.prompt_task_examples == "random":
        random.seed(args.seed)
        prompt_egs_keys = random.sample(list(prompt_egs.keys()), args.prompt_num_examples)

        for eg in prompt_egs_keys:
            prompt += "\n\n" + prompt_egs[eg]

    # abalation settings
    ## 얘는 옵션인듯
    ## prompt에서 주석 제거
    if args.prompt_task_examples_ablation == "no_comments":
        prompt = prompt.split('\n')
        prompt = [line for line in prompt if "# " not in line]
        prompt  = "\n".join(prompt)

    ## prompt에서 assert 문 제거
    if args.prompt_task_examples_ablation == "no_feedback":
        prompt = prompt.split('\n')
        prompt = [line for line in prompt if not any([x in line for x in ["assert", "else"]])]
        prompt  = "\n".join(prompt)

    ## prompt에서 둘 다 제거
    if args.prompt_task_examples_ablation == "no_comments_feedback":
        prompt = prompt.split('\n')
        prompt = [line for line in prompt if not any([x in line for x in ["assert", "else", "# "]])]
        prompt  = "\n".join(prompt)


    ## 여기까지는 prompt에 0으로 초기화 된 환경에 할 수 있는 행동, 있는 obj, 예시의 미리 정해진 plan 넣어줬음.
    
    ##################

    ## 평가 구문. 기존환경/새로운 환경 나뉘어져있음
    # evaluate in given unseen env
    ## virtualhome 에서 env_id가 있음. 방 구조 다름
    ## 새로운 환경(unseen)에 대한 obj prompt에 추가
    if args.env_id != 0:
        comm.reset(args.env_id)
        _, graph = comm.environment_graph()
        obj = list(set([node['class_name'] for node in graph["nodes"]]))
        prompt += f"\n\n\nobjects = {obj}"

        # evaluation tasks in given unseen env
        test_tasks = []
        with open(f"{args.progprompt_path}/data/new_env/{args.test_set}_annotated.json", 'r') as f:
            for line in f.readlines():
                test_tasks.append(list(json.loads(line).keys())[0])
        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")

    # setup logging
    log_filename = f"{args.expt_name}_{args.prompt_task_examples}_{args.prompt_num_examples}examples"
    if args.prompt_task_examples_ablation != "none":
        log_filename += f"_{args.prompt_task_examples_ablation}"
    log_filename += f"_{args.test_set}"
    log_file = open(f"{args.progprompt_path}/results/{log_filename}_logs.txt", 'w')
    log_file.write(f"\n----PROMPT for planning----\n{prompt}\n")
    
    # evaluate in seen env
    if args.env_id == 0:
        test_tasks = []
        for file in os.listdir(f"{args.progprompt_path}/data/{args.test_set}"):
            with open(f"{args.progprompt_path}/data/{args.test_set}/{file}", 'r') as f:
                for line in f.readlines():
                    test_tasks.append(list(json.loads(line).keys())[0])

        log_file.write(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")

    # test_tasks = test_tasks[:3] ## debug to check changes
    ## prompt에 0으로 초기화 된 환경에 할 수 있는 행동, 있는 obj, 예시의 미리 정해진 plan, unseen에 대한 object를 넣어줌
    # generate plans for the test set
    if not args.load_generated_plans:
        gen_plan = []
        for task in test_tasks:
            print(f"Generating plan for: {task}\n")
            prompt_task = "def {fxn}():".format(fxn = '_'.join(task.split(' '))) # task _로 이어져있는거 공백으로 잇기
            curr_prompt = f"{prompt}\n\n{prompt_task}\n\t" ## 주어진 정보 + 수행할 task 이어서 prompt 만듦
            _, text = LM(curr_prompt, 
                        args.gpt_version, 
                        max_tokens=600, 
                        stop=["def"], 
                        frequency_penalty=0.15)
            gen_plan.append(text) # 답장온거 저장
            # because codex has query limit per min
            if args.gpt_version == 'code-davinci-002':
                time.sleep(90)

        # save generated plan
        line = {}
        print(f"Saving generated plan at: {log_filename}_plans.json\n")
        with open(f"{args.progprompt_path}/results/{log_filename}_plans.json", 'w') as f:
            for plan, task in zip(gen_plan, test_tasks):
                line[task] = plan
            json.dump(line, f)

    # load from file
    else:
        print(f"Loading generated plan from: {log_filename}.json\n")
        with open(f"{args.progprompt_path}/results/{log_filename}_plans.json", 'r') as f:
            data = json.load(f)
            test_tasks, gen_plan = [], []
            for k, v in data.items():
                test_tasks.append(k)
                gen_plan.append(v)

    
    log_file.write(f"\n----PROMPT for state check----\n{current_state_prompt}\n")

    # run execution
    print(f"\n----Runing execution----\n")
    final_states, initial_states, exec_per_task = run_execution(args, 
                                                                comm, 
                                                                test_tasks, 
                                                                gen_plan,
                                                                log_file)
    

    #evaluate
    final_states_GT = []
    with open(f'{args.progprompt_path}/data/final_states/final_states_{args.test_set}.json', 'r') as f:
        for line in f.readlines():
            final_states_GT.append((json.loads(line)))

    results = eval(final_states, 
         final_states_GT, 
         initial_states, 
         test_tasks,
         exec_per_task,
         log_file)

    print(f"\n----Results----\n{results['overall']}\n")
    with open(f"{args.progprompt_path}/results/{log_filename}_metric.json", 'w') as f:
        json.dump(results, f)
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--progprompt-path", type=str, required=True)
    parser.add_argument("--expt-name", type=str, required=True)

    parser.add_argument("--openai-api-key", type=str, 
                        default="sk-xyz")
    parser.add_argument("--unity-filename", type=str, 
                        default="/path/to/macos_exec.v2.3.0.app")
    parser.add_argument("--port", type=str, default="8000")
    parser.add_argument("--display", type=str, default="0")
    
    parser.add_argument("--gpt-version", type=str, default="text-davinci-002", 
                        choices=['text-davinci-002', 'davinci', 'code-davinci-002'])
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--test-set", type=str, default="test_unseen", 
                        choices=['test_unseen', 'test_seen', 'test_unseen_ambiguous', 'env1', 'env2'])

    parser.add_argument("--prompt-task-examples", type=str, default="default", 
                        choices=['default', 'random'])
    # for random task examples, choose seed
    parser.add_argument("--seed", type=int, default=0)
    
    ## NOTE: davinci or older GPT3 versions have a lower token length limit
    ## check token length limit for models to set prompt size: 
    ## https://platform.openai.com/docs/models
    parser.add_argument("--prompt-num-examples", type=int, default=3, 
                         choices=range(1, 7))
    parser.add_argument("--prompt-task-examples-ablation", type=str, default="none", 
                         choices=['none', 'no_comments', "no_feedback", "no_comments_feedback"])

    parser.add_argument("--load-generated-plans", type=bool, default=False)
    
    args = parser.parse_args()
    openai.api_key = args.openai_api_key

    if not osp.isdir(f"{args.progprompt_path}/results/"):
            os.makedirs(f"{args.progprompt_path}/results/")

    planner_executer(args=args)