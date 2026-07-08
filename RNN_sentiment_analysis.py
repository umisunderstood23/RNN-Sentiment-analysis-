#This case study will demonstrate sentiment analysis using recurrent neural network
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import shutil
import numpy as np
def tokenizationDemo(data):
    token_list=[]
    for i in data:
        a=i.split()
        for word in a:
            if word not in token_list:
                token_list.append(word)
            else:
                pass
    token_dic={}
    tokens=1
    for i in token_list:
        token_dic[i]=tokens
        tokens+=1
    print("Here are your words in data with unique tokens: ",token_dic)



def tokenization_training(data):
    object=Tokenizer(num_words=50)
    object.fit_on_texts(data)
    tokenized_data=object.texts_to_sequences(data)
    print(object.word_index)      # Shows unique words and their assigned ID numbers
    print("Below is the token transformed data")
    print(tokenized_data)
    '''
    Explanation: We created a tokenizer object first. Then used fit_to_texts
    This makes a internal dictionary map having each unique word with their each unique token.
    Then using texts_to_sequences uses this created dictionary to find the words and actually change the sentence data by inputting tokens for each word
    '''

        #Padding data 
    padded_data=pad_sequences(tokenized_data,padding="post")
    print("Here is your data padded: ")
    print("\n\n")
    print(padded_data)
    return object,padded_data

    






def main():
    terminal_width=shutil.get_terminal_size().columns

    border="="*180
    print(border)
    print(border)
    print("Mandar's sentiment predictor".center(terminal_width))
    print("\n\n")
    print("Overview: This code uses tokenization and embeddings sort of concepts on a simple RNN layer to predict sentence sentiments".center(terminal_width))
    print(border)
    print(border)

    #sentences data 
    data= sentences = [
        "I love this movie",
        "This film was great",
        "What a fantastic experience",
        "I really enjoyed it",
        "Absolutely wonderful acting",
        "I hate this movie",
        "This film was terrible",
        "What a bad experience",
        "I really disliked it",
        "Absolutely horrible acting"
    ]
    labels=[1,1,1,1,1,0,0,0,0,0]
    tokenizer,padded_data=tokenization_training(data=data)
    print("Number of sentences in data are: ", len(padded_data))
    print("So the number of labels are: ", len(labels))
    


    X=np.array(padded_data)
    Y=np.array(labels)
    

    ##Create a custom RNN model
    model=Sequential()
    model.add(Embedding(input_dim=50,output_dim=8,input_length=8))
    model.add(SimpleRNN(16,activation="tanh"))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    print(model.summary())
    

    #Train model 
    model.fit(X,Y,epochs=30,verbose=0)
    

    test_sentence=input("Enter your sentence to test the sentiment analysis model on: ")
    test_sentence=[test_sentence]
    for_final=test_sentence
    test_sentence=tokenizer.texts_to_sequences(test_sentence)
    test_sentence=pad_sequences(test_sentence,padding="post")
    #Test model
    answer=model.predict(test_sentence)
    print(answer)
    if answer>0.5:
        print(f"Sentence entered: {for_final}\nModel's sentiment prediction: ""Positive""")
    else:
        
     print(f"Sentence entered: {for_final}\nModel's sentiment prediction: ""Negative"" ")


if __name__=="__main__":
    main()