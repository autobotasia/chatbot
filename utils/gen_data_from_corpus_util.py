import json
import jsonlines
import gzip
import numpy as np

output_file_name = "data.json"

def saveJson(datastore):  
  # Writing JSON data
  with open(output_file_name, 'w') as f:
      json.dump(datastore, f)

def gen_data():
  number_of_lines = 0
  current_dialog_source = ''
  current_dialog_target = ''
  dialog_silenced = False
  retval = []
  # Iterate through the file and build list of dialogs separated by __eou__.
  for line in open('train_none_original_no_cands.txt', errors='ignore'):
    if number_of_lines % 10000 == 0:
      print('Parsed ' + str(number_of_lines) + ' lines.')

    dialog_id = line.split()[0]
    # Check if this is a refurbished line.
    if ('__SILENCE__' not in line and
            ((dialog_silenced and dialog_id == '1') or not dialog_silenced)):
      dialog_silenced = False
      number_of_lines += 1

      # Get the utterances.
      current_dialog_source = ' '.join(line.split('\t')[0].split()[1:])
      current_dialog_target = line.split('\t')[1].strip('\n')
      
      context = ''

      # Whether this is a new dialog.
      if dialog_id == '1':
        if context == '':
          context = current_dialog_source + ' ' + current_dialog_target + ' '
        retval.append({
          'input': current_dialog_source,
          'target': current_dialog_target,
          'context': context
        })

        context = ''
      else:
        context += context + current_dialog_source + ' ' + current_dialog_target + ' '
        retval.append({
          'input': current_dialog_source,
          'target': current_dialog_target,
          'context': context
        })

    else:
      dialog_silenced = True

  saveJson(retval)  

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    jsdata = eval(l)    
    yield {
        'input': jsdata["question"],
        'target': jsdata["answer"],
        'context': jsdata["question"] + ' ' + jsdata["answer"]
      }

def amazon_qa():
  retval = []
  with open(output_file_name, 'r') as f:
    retval = json.load(f)
  filenames = [
    "qa_Appliances.json.gz",
    "qa_Arts_Crafts_and_Sewing.json.gz",
    "qa_Automotive.json.gz",
    "qa_Baby.json.gz",
    "qa_Beauty.json.gz",    
    "qa_Cell_Phones_and_Accessories.json.gz",
    "qa_Clothing_Shoes_and_Jewelry.json.gz",
    "qa_Electronics.json.gz",
    "qa_Health_and_Personal_Care.json.gz",
    "qa_Home_and_Kitchen.json.gz",
    "qa_Industrial_and_Scientific.json.gz",
    "qa_Musical_Instruments.json.gz",
    "qa_Office_Products.json.gz",
    "qa_Patio_Lawn_and_Garden.json.gz",
    "qa_Pet_Supplies.json.gz",
    "qa_Software.json.gz",
    "qa_Sports_and_Outdoors.json.gz",
    "qa_Tools_and_Home_Improvement.json.gz",
    "qa_Toys_and_Games.json.gz",
    "qa_Video_Games.json.gz"
  ]
  
  for filename in filenames:
    for l in parse("./amazon_qa/"+filename):
      retval.append(l)
  saveJson(retval)  


def squad_qa():
  retval = []
  #with open(output_file_name, 'r') as f:
  #  retval = json.load(f)


  with open("train-v1.1.json", 'r') as fp:
    squad = json.load(fp)

  version = squad["version"]
  for article in squad["data"]:
    if "title" in article:
      title = article["title"].strip()
    else:
      title = "no title"
    for paragraph in article["paragraphs"]:
      context = paragraph["context"].strip()
      for qa in paragraph["qas"]:
        question = qa["question"].strip()
        id_ = qa["id"]
        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
        answers = [answer["text"].strip() for answer in qa["answers"]]

        # Features currently used are "context", "question", and "answers".
        # Others are extracted here for the ease of future expansions.
        example = {
            #"version": version,
            #"title": title,
            "input": question,
            "target": answers[0],
            "context": context,            
            #"id": id_,
            #"answer_starts": answer_starts,
            
            #"num_answers": len(answers),
            #"is_supervised": True,
        }
        retval.append(example)
  
  saveJson(retval)  

def dailychat():
  retval = []
  with open(output_file_name, 'r') as f:
    retval = json.load(f)


  dialogs = open('./dailychat/dialogues_text.txt', errors='ignore')
  for dialog in dialogs:    
    # Utterances are separated by the __eou__ token.
    utterances = dialog.split('__eou__')[:-1]
    loop = 0
    while(loop < len(utterances) - 1):
      tmpdata = {
        "input"   : utterances[loop],
        "target"  : utterances[loop+1],
        "context" : ' '.join(utterances)
      }        
      retval.append(tmpdata)
      loop +=2        
  #print(retval)    
  saveJson(retval)  

def natural_questions():
  filenames = [
    'nq-train-00.jsonl.gz',
    'nq-train-17.jsonl.gz',
    'nq-train-34.jsonl.gz',
    'nq-train-01.jsonl.gz',  
    'nq-train-18.jsonl.gz',  
    'nq-train-35.jsonl.gz',
    'nq-train-02.jsonl.gz',  
    'nq-train-19.jsonl.gz',  
    'nq-train-36.jsonl.gz',
    'nq-train-03.jsonl.gz',  
    'nq-train-20.jsonl.gz',  
    'nq-train-37.jsonl.gz',
    'nq-train-04.jsonl.gz',  
    'nq-train-21.jsonl.gz',  
    'nq-train-38.jsonl.gz',
    'nq-train-05.jsonl.gz',  
    'nq-train-22.jsonl.gz',  
    'nq-train-39.jsonl.gz',
    'nq-train-06.jsonl.gz',  
    'nq-train-23.jsonl.gz',  
    'nq-train-40.jsonl.gz',
    'nq-train-07.jsonl.gz',  
    'nq-train-24.jsonl.gz',  
    'nq-train-41.jsonl.gz',
    'nq-train-08.jsonl.gz',  
    'nq-train-25.jsonl.gz',  
    'nq-train-42.jsonl.gz',
    'nq-train-09.jsonl.gz',  
    'nq-train-26.jsonl.gz',  
    'nq-train-43.jsonl.gz',
    'nq-train-10.jsonl.gz',  
    'nq-train-27.jsonl.gz',  
    'nq-train-44.jsonl.gz',
    'nq-train-11.jsonl.gz',  
    'nq-train-28.jsonl.gz',  
    'nq-train-45.jsonl.gz',
    'nq-train-12.jsonl.gz',  
    'nq-train-29.jsonl.gz',  
    'nq-train-46.jsonl.gz',
    'nq-train-13.jsonl.gz',  
    'nq-train-30.jsonl.gz',  
    'nq-train-47.jsonl.gz',
    'nq-train-14.jsonl.gz',  
    'nq-train-31.jsonl.gz',  
    'nq-train-48.jsonl.gz',
    'nq-train-15.jsonl.gz',  
    'nq-train-32.jsonl.gz',  
    'nq-train-49.jsonl.gz',
    'nq-train-16.jsonl.gz',  
    'nq-train-33.jsonl.gz'
  ]

  for filename in filenames:
    g = gzip.open('./natural_questions/v1.0/train/' + filename, 'r')
    for l in g:
      obj = json.loads(l)      
      document_html = obj['document_html'].encode('utf-8')      
      question_text = obj['question_text']
      context = ''
      answer_text = ''      
      anno = obj['annotations'][0]
      has_long_answer = anno['long_answer']['start_byte'] >= 0
      has_short_answer = anno['short_answers'] or anno['yes_no_answer'] != 'NONE'
      long_answers = [
        a['long_answer']
        for a in obj['annotations']
        if a['long_answer']['start_byte'] >=0
      ]
      short_answers = [
        a['short_answers']
        for a in obj['annotations']
        if a['short_answers'] and has_short_answer
      ]
      
      yes_no_answers = [
          a['yes_no_answer']
          for a in obj['annotations']
          if a['yes_no_answer'] != 'NONE' and has_short_answer
      ]

      if has_long_answer and has_short_answer:
        long_answer_bounds = [
          (la['start_byte'], la['end_byte']) for la in long_answers
        ]
        long_answer_counts = [
            long_answer_bounds.count(la) for la in long_answer_bounds
        ]
        long_answer = long_answers[np.argmax(long_answer_counts)]
        context = document_html[long_answer["start_byte"]:long_answer["end_byte"]]
      
        short_answers_ids = [[
            (s['start_byte'], s['end_byte']) for s in a
        ] for a in short_answers] + [a for a in yes_no_answers]
        short_answers_counts = [
            short_answers_ids.count(a) for a in short_answers_ids
        ]
        
        short_answers_texts = [
            ', '.join([
                "%s"%document_html[s['start_byte']:s['end_byte']]
                for s in short_answer
            ])
            for short_answer in short_answers
        ]

        short_answers_texts += yes_no_answers
        answer_text = short_answers_texts[np.argmax(short_answers_counts)]

        yield {
          "input"   : question_text,
          "target"  : answer_text,
          "context" : context
        } 
        

  
#natural_questions()
#gen_data()
#amazon_qa()
squad_qa()
#dailychat()