import json
import random

from openai_access import call_chatgpt
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt


fr = open('alpaca_data_cleaned.json','r')

all_objs = json.load(fr)

evol_objs = []


for cur_obj in all_objs:
	
	instruction = cur_obj['instruction'].strip() + '\r\n'+ cur_obj['input'].strip()

	evol_prompts = []
	evol_prompts.append(createConstraintsPrompt(instruction))
	evol_prompts.append(createDeepenPrompt(instruction))
	evol_prompts.append(createConcretizingPrompt(instruction))
	evol_prompts.append(createReasoningPrompt(instruction))
	evol_prompts.append(createBreadthPrompt(instruction))

	selected_evol_prompt = random.choice(evol_prompts)


	evol_instruction = call_chatgpt(selected_evol_prompt)
	answer = call_chatgpt(evol_instruction)

	evol_objs.append({"instruction":evol_instruction,"output":answer})



with open('alpaca_data_evol.json', 'w') as f:	
	json.dump(evol_objs, f, indent=4)




