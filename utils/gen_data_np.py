import json
import collections
import gzip
import numpy as np
import tensorflow as tf

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

TextSpan = collections.namedtuple("TextSpan", "token_positions text")  
flags = tf.flags
FLAGS = flags.FLAGS

def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
  if (e["long_answer_candidates"][idx]["top_level"]):
    return True
  elif not get_candidate_text(e, idx).text.strip():
    # Skip empty contexts.
    return True
  else:
    return False


def get_first_annotation(e):
  """Returns the first short or long answer in the example.

  Args:
    e: (dict) annotated example.

  Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
  """
  positive_annotations = sorted(
      [a for a in e["annotations"] if has_long_answer(a)],
      key=lambda a: a["long_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["short_answers"]:
      idx = a["long_answer"]["candidate_index"]
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][-1]["end_token"]
      return a, idx, (token_to_char_offset(e, idx, start_token),
                      token_to_char_offset(e, idx, end_token) - 1)

  for a in positive_annotations:
    idx = a["long_answer"]["candidate_index"]
    return a, idx, (-1, -1)

  return None, -1, (-1, -1)

def get_text_span(example, span):
  """Returns the text in the example's document in the given token span."""
  token_positions = []
  tokens = []
  for i in range(span["start_token"], span["end_token"]):
    t = example["document_tokens"][i]
    if not t["html_token"]:
      token_positions.append(i)
      token = t["token"].replace(" ", "")
      tokens.append(token)
  return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
  return char_offset


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "Table"
  elif first_token == "<P>":
    return "Paragraph"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "List"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "Other"
  else:
    tf.logging.warning("Unknoww candidate type found: %s", first_token)
    return "Other"


def add_candidate_types_and_positions(e):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e):
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < 50:
      counts[context_type] += 1
    c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return TextSpan([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])


def candidates_iter(e):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx):
      continue
    yield idx, c

def create_example_from_jsonl(line):
  """Creates an NQ example from a given line of JSON."""
  e = json.loads(line, object_pairs_hook=collections.OrderedDict)
  add_candidate_types_and_positions(e)
  annotation, annotated_idx, annotated_sa = get_first_annotation(e)

  # annotated_idx: index of the first annotated context, -1 if null.
  # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
  question = {"input_text": e["question_text"]}
  answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "input_text": "long",
  }

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found.
  if annotated_sa != (-1, -1):
    answer["input_text"] = "short"
    span_text = get_candidate_text(e, annotated_idx).text
    answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
    answer["span_start"] = annotated_sa[0]
    answer["span_end"] = annotated_sa[1]
    expected_answer_text = get_text_span(
        e, {
            "start_token": annotation["short_answers"][0]["start_token"],
            "end_token": annotation["short_answers"][-1]["end_token"],
        }).text
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= 48:
      break

  # Assemble example.
  example = {
      "name": e["document_title"],
      "id": str(e["example_id"]),
      "questions": [question],
      "answers": [answer],
      "has_correct_context": annotated_idx in context_idxs
  }

  single_map = []
  single_context = []
  offset = 0
  for context in context_list:
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
    offset += len(single_context[-1]) + 1
    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])

  return example  

def natural_questions():
  #input_paths = tf.gfile.Glob("./tiny-dev/nq-train-??.jsonl.gz")
  input_paths = tf.gfile.Glob("./tiny-dev/nq-dev-sample.jsonl.gz")
  input_data = []

  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.gfile.Open(path, "rb"))
    else:
      return tf.gfile.Open(path, "rb")

  for path in input_paths:
    retval = []
    with _open(path) as input_file:
      for line in input_file:
        entry = create_example_from_jsonl(line)
        answer = entry['answers'][0]['span_text']
        if answer == '':
            answer = entry['name']
        retval.append({
        "input"   : entry['questions'][0]['input_text'],
        "target"  : answer,
        "context" : entry['name']
        })
        with open(path[:-8] + 'json', 'w') as f:
            json.dump(retval, f)    
      

  
natural_questions()
#gen_data()
#amazon_qa()
#squad_qa()
#dailychat()