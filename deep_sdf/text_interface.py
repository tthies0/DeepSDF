from transformers import pipeline

def text_interpreter():
    class_names = ["airplane",
                   "bench",
                   "car",
                   "chair",
                   "lamp",
                   "rifle",
                   "sofa",
                   "table",
                   "watercraft"
                   ]
    print(class_names)
    classifier = pipeline("zero-shot-classification", model="knowledgator/comprehend_it-base")
    print("----------------------------")
    print("Please input word: ")
    input_text = input()
    result = classifier(input_text, candidate_labels=class_names)
    print(result)

text_interpreter()