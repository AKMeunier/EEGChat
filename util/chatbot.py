from typing import Optional, List

import openai


with open('../api_keys/openai.txt', 'r') as f:
    openai.api_key = f.read()


def generate_answer(question, keyword):
    """
    Generate a full sentence answer for a given question and keyword using GPT3
    :param question: str; question
    :param keyword: str; keyword
    :return: str; full sentence answer
    """
    prompt = "The following is a telephone conversation. Write one sentence as the next response " \
             "using the keyword: {}. Do not add any additional details.\n" \
             "\n" \
             "\"{}\"\n" \
        .format(keyword.lower(), question)

    again=True
    while again:
        try:
            output = openai.Completion.create(engine="text-davinci-003",
                                              prompt=prompt,
                                              max_tokens=50,
                                              temperature=0.7
                                              )
            again = False
        except openai.error.OpenAIError as e:
            print(e)
            again = True

    answer = output['choices'][0]['text']
    return answer.replace('"', '').replace('\n', '')


def generate_answer_finetuned(question, keyword, history=""):
    """
    Generate a full sentence answer for a given question and keyword using a fine-tuned GPT3 model.
    :param question: str; question
    :param keyword: str; keyword
    :return: str; full sentence answer
    """

    # model trained on "high quality" version of dataset
    model_id = "davinci:ft-speech-bci:hq-2023-03-20-14-26-28"

    prompt = history + "Question: {}\n" \
             "Keywords: {}\n" \
             "Answer:\n\n###\n\n".format(question, keyword)

    output = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        temperature=0.7,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Answer:", "Keywords:", "Question:", " END"]
    )

    answer = output['choices'][0]['text'].replace('"', '').replace('\n', '')

    while answer[0] in [' ', '\n']:
        answer = answer[1:]

    return answer


def generate_keywords(question: str, n: int, knowledge_base=None):
    """
    Generate n keywords for a given question using GPT3
    :param question: str; question
    :param n: int; number of keywords
    :return: [str]; list of keywords
    """
    prompt = "Generate {} different one word answers to the question: \"{}\". " \
             "Make the answers as different as possible.\n Answers: ".format(n, question)

    try_again = True
    while try_again:
        output = openai.Completion.create(engine="text-davinci-003",
                                          prompt=prompt,
                                          max_tokens=256,
                                          temperature=0.5
                                          )

        keywords, try_again = process_keywords(output=output, n=n)

    return keywords


def generate_keywords_yes_no_fork(question: str, n: int, knowledge_base=None):
    """
    Generate n keywords for a given question using GPT3, first evaluating whether the question is a yes-no question.
    :param question: str; question
    :param n: int; number of keywords
    :return: [str]; list of keywords
    """
    prompt_yes_no = "Is 'yes' a grammatically correct answer to the following question? " \
                    "Question: {}".format(question)

    output_yes_no = openai.Completion.create(engine="text-davinci-003",
                                             prompt=prompt_yes_no,
                                             max_tokens=5,
                                             temperature=0.5
                                             )

    yes_no_question = output_yes_no['choices'][0]['text'].replace('\n', '')[:3]\
                          .replace('.', '')\
                          .replace(',','')\
                          .replace('!', '')\
                          .lower() == 'yes'

    if yes_no_question:
        prompt = "Generate {} different one word answers to the question: \"{}\". " \
                 "Include affirmative and negative answers in the options. \n Answers: ".format(n, question)
    else:
        prompt = "Generate {} different one word answers to the question: \"{}\". " \
                 "Make the answers as different as possible. \n Answers: ".format(n, question)

    try_again = True
    while try_again:
        output = openai.Completion.create(engine="text-davinci-003",
                                          prompt=prompt,
                                          max_tokens=256,
                                          temperature=0.5
                                          )

        keywords, try_again = process_keywords(output=output, n=n)

    return keywords


def generate_keywords_finetune(question: str, n: int, history: Optional[List[str]] = None, knowledge_base=None):
    """Generates n keywords to the given question using a finetuned gpt3 model.
    :param question: str; question
    :param n: int; number of keywords
    :param history: Optional[List[str]]; conversation history, where each entry is one utterance. Has to be ordered,
      where index 0 contains the oldest utterance.
    :return: [str]; list of keywords
    """
    prompt = ""
    if history is not None:
        assert len(history) % 2 == 0
        for i in range(0, len(history), 2):
            prompt += f"Question: {history[i]}\nAnswer: {history[i+1]}\n"

    prompt += f"Question: {question}\nAnswer count: {n}\nAnswer:\n\n###\n\n"
    model_name = "davinci:ft-speech-bci:keyword-generator-v3-2023-05-10-11-43-44"

    try_again = True
    keywords = None
    # the while loop is just a safety mechanism should the output of the prompt fail for some reason
    while try_again:
        output = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=0.5,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["END"]
        )

        keywords, try_again = process_keywords(output=output, n=n)

    assert keywords is not None
    return keywords


def process_keywords(output, n: int):
    """
    Process output openai models into a list of n keywords. If try_again is True, the keyword generation should be
    repeated.
    :param output: return of openai.Completion.create
    :param n: number of keywords that the text in output should contain
    :return: ([str], bool); keywords, try_again
    """
    kw_list = output['choices'][0]['text'][1:]\
        .replace('.', '')\
        .replace('?', '')\
        .replace('!', '')\
        .replace(', \n', '\n')\
        .replace(', ', '\n')\
        .split('\n')

    if len(kw_list) == 1:
        kw_list = kw_list[0].split(' ')

    keywords = []
    for kw in kw_list:
        if len(kw) == 0:
            continue

        this_kw = kw.split(' ')
        if '' in this_kw:
            this_kw.remove('')
        this_kw = this_kw[-1].replace(',', '').capitalize()
        keywords.append(this_kw)

    try_again = False

    if len(keywords) > n:
        keywords = keywords[:n]
    elif len(keywords) < n:
        try_again = True

    return keywords, try_again


def generate_keywords_details(question: str, n: int, knowledge_base=None):
    """
    Generate n keywords for a given question using GPT3
    :param question: str; question
    :param n: int; number of keywords
    :return: [str]; list of keywords
    """
    prompt = f"Generate N keywords that might help a speech-impaired person respond to a given question. The keywords " \
             f"should be as short as possible and only describe one possible answer each. Provide answers which are as " \
             f"different as possible and try to include every viewpoint in the answers. For example if one " \
             f"of the answers is yes, also include no, and when one of the answers is good, also include bad. When " \
             f"the question is asking for a day or time, be specific in your suggested answers. " \
             f"In addition to suggesting answers, also provide the category of what the question is asking for. For " \
             f"example, if the question is asking for the name of a person, the category should be NAME. " \
             f"If the question is asking for an address or street name, the category should be ADDRESS. Here are " \
             f"some examples:\n\n" \
             "\n" \
             "Example 1:\n" \
             "Question: How was your day?\n" \
             "N: 6\n" \
             "Answers: 1. Good; 2. Fantastic; 3. Bad; 4. Horrible; 5. Splendid; 6. Boring\n" \
             "Category: ADJECTIVE\n" \
             "Example 2:\n" \
             "Question: How many people are living in your household?\n" \
             "N: 10\n" \
             "Answers: 1. 1; 2. 2; 3. 3; 4. 4; 5. 5; 6. 6; 7. 7; 8. 8; 9. 9; 10. 10\n" \
             "Category: NUMBER\n" \
             "Example 3:\n" \
             "Question: What is your mother's name?\n" \
             "N: 4\n" \
             "Answers: 1. Rose; 2. Mary; 3. Miriam; 4. Joanna\n" \
             "Category: NAME\n" \
             "Example 4:\n" \
             "Question: Are you hungry?\n" \
             "N: 3\n" \
             "Answers: 1. Yes; 2. No; 3. Very\n" \
             "Category: YESNO\n" \
             "\n" \
             f"Question: {question}\n" \
             f"N: {n}\n"

    while True:
        again = True
        while again:
            try:
                output = openai.Completion.create(engine="text-davinci-003",
                                                  prompt=prompt,
                                                  max_tokens=256,
                                                  temperature=0.5
                                                  )['choices'][0]['text']
                again = False
            except openai.error.OpenAIError as e:
                print(e)
                again = True

        output_list = output.split("Category: ")
        category = output_list[1]
        print(category)

        if knowledge_base is not None:
            for k in knowledge_base.keys():
                 if category == k:
                     return knowledge_base[k][:n] + max(0, n - len(knowledge_base[k])) * ['-']

        keywords1 = [kw.split(".")[1] for kw in output_list[0][8:-1].split(";")]
        keywords = [kw[1:] if kw[0] == ' ' else kw for kw in keywords1]

        if len(keywords) >= n:
            return keywords[:n]


if __name__ == "__main__":
    questions = ["What's your name?",
                 "What name is the reservation for?",
                 "What is your favorite food?",
                 "Would you like to eat something?",
                 "What time are you eating breakfast?",
                 "How many people would you like to make the reservation for?",
                 "What is your address?",
                 "Pizzeria romano how can I help you",
                 "Fitness studio get fit how can I help you?",
                 "Cafe Marie how can I help you?",
                 "911 what's your emergency?",
                 "Practice doctor robinson how can I help you?"
                 ]

    n = 12

    for question in questions:
        kw = generate_keywords_details(question, n)
        a = generate_answer_finetuned(question, kw[1])
        print("Q: " + question)
        print(kw)
        #print("KW: " + kw[0])
        print("A:  " + a + "\n")