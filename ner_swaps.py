'''
authors: saurav.pattnaik & srishti.verma :)
'''

from similarity_module import prepare_freq_table,tokenizer
import heapq
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def NER_tags_creator(sentence, model):
    ner_results = model(sentence)
    token = None
    tag = None
    ner_pairs = []
    i = 0

    while i < len(ner_results):
        if tag is None:
            token = ner_results[i]['word']
            tag = ner_results[i]['entity'][2:]

        elif 'B' in ner_results[i]['entity']:
            ner_pairs.append((tag, token))
            token = ner_results[i]['word']
            tag = ner_results[i]['entity'][2:]

        else:
            token += ' ' + ner_results[i]['word']
            token = re.sub(r'[#]+', '', token)
        i += 1

    ner_pairs.append((tag, token))
    ner_pairs = list(set(ner_pairs))
    return ner_pairs


def build_list_for_heap(pairs, freq_table):
    person_heap = []
    org_loc_heap = []

    for tag, word in pairs:
        freq = -1 * freq_table.get(word, 0)
        if tag == "PER":
            person_heap.append((freq, word))

        elif tag in ['LOC', 'ORG']:
            org_loc_heap.append((freq, word))

    return person_heap, org_loc_heap


def generate_swappers(s1, s2, model,
                      freq_table):

    s1_ner_pairs = NER_tags_creator(s1, model)
    s2_ner_pairs = NER_tags_creator(s2, model)

    person_heap_s1, org_loc_heap_s1 = build_list_for_heap(s1_ner_pairs, freq_table)
    person_heap_s2, org_loc_heap_s2 = build_list_for_heap(s2_ner_pairs, freq_table)

    heapq.heapify(person_heap_s1)
    heapq.heapify(person_heap_s2)

    heapq.heapify(org_loc_heap_s1)
    heapq.heapify(org_loc_heap_s2)

    new_sentences = []

    if len(person_heap_s1) != 0 and len(person_heap_s2) != 0:
        word_s1 = heapq.heappop(person_heap_s1)[1]
        word_s2 = heapq.heappop(person_heap_s2)[1]

        new_sentences.append(s1.replace(word_s1, word_s2))
        new_sentences.append(s2.replace(word_s2, word_s1))

    if len(org_loc_heap_s1) != 0 and len(org_loc_heap_s2) != 0:
        word_s1 = heapq.heappop(org_loc_heap_s1)[1]
        word_s2 = heapq.heappop(org_loc_heap_s2)[1]

        new_sentences.append(s1.replace(word_s1, word_s2))
        new_sentences.append(s2.replace(word_s2, word_s1))

    return new_sentences

# THIS IS THE CALLING FUNCTION
def create_data_plus_ner_swap(corpus1, corpus2, sm_df, source, target, value_name, threshold=0.7):

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    freq_mapper = prepare_freq_table(corpus1, corpus2)
    sm_df = sm_df[sm_df[value_name] > threshold].sort_values(by=source)
    final_data = []
    ner_samples = []
    count_swaps = 0
    for i, j in zip(sm_df[source], sm_df[target]):
        s1 = corpus1[i]
        s2 = corpus2[j]
        samples = generate_swappers(s1, s2, nlp, freq_mapper)
        if len(samples) > 0:
            count_swaps += len(samples)
        final_data.append(s1)  # S1 sentence
        final_data.append(s2)  # S2 sentence
        final_data.extend(samples)  # NER Swaps added
        ner_samples.extend(samples)  # All NER swaps samples

    ner_samples = [s for s in ner_samples if s not in corpus1 and s not in corpus2]
    return final_data, ner_samples, count_swaps