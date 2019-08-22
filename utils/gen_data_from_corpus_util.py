import json
import gzip

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
      
      # Whether this is a new dialog.
      if dialog_id == '1' and current_dialog_source != '':
        retval.append({
          'input': current_dialog_source,
          'target': current_dialog_target,
          'context': current_dialog_source
        })                  
      else:
        current_dialog_source += current_dialog_source + '\n'
        current_dialog_target += current_dialog_target + '\n'

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
        'context': jsdata["question"]
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
  with open(output_file_name, 'r') as f:
    retval = json.load(f)


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
      tmpdata = {
        "input"   : ' '.join(utterances),
        "target"  : ' '.join(utterances[1:]),
        "context" : ' '.join(utterances)
      }
      retval.append(tmpdata)
  #print(retval)    
  saveJson(retval)  




gen_data()
amazon_qa()
squad_qa()
dailychat()