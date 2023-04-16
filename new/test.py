import spacy
from spacy.util import minibatch, compounding

def train_ner(train_data: list, labels: list, n_iter: int = 10):
    # Load a blank English model
    nlp = spacy.blank("en")
    
    # Create a new NER pipe and add it to the pipeline
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner)
    
    # Add the custom labels to the NER pipe
    for label in labels:
        ner.add_label(label)
    
    # Start the training
    nlp.begin_training()
    
    # Train for n_iter iterations
    for itn in range(n_iter):
        losses = {}
        
        # Shuffle the training data
        random.shuffle(train_data)
        
        # Batch the examples and iterate over them
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, losses=losses)
        
        print(f"Losses at iteration {itn}: {losses}")
    
    return nlp

# Example usage:
train_data = [
  ("I like London", {"entities": [(7, 13, "GPE")]}),
  ("I'm from New York", {"entities": [(9, 17 , "GPE")]})
]
labels = ["GPE"]

ner_model = train_ner(train_data=train_data,
                      labels=labels,
                      n_iter=10)

doc = ner_model("I live in Paris")
for ent in doc.ents:
    print(ent.text, ent.label_)