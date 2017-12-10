Program run with either "train" or "test" mode
1. Train mode:
    words_prediction_lstm.py  <mode> <data_file> <model_file> <max_update> <learning_rate>
    <mode>          : train
    <data_file>     : training data file
    <model_file>    : model file (to save optimized paramaters of the model after training)
    <max_update>    : maximum number of update
    <learning_rate> :

    For example: words_prediction_lstm.py train input.txt model_file 5000 0.001

2. Test mode:
    words_prediction_lstm <mode> <data_file> <model_file> <sample_text> <newtext_length>
    <mode>              : test
    <data_file>         : data file (to check new generated text)
    <model_file>        : model file (saved from train step)
    <sample_text>       : feed the first text for LSTM to generate new words (by default,
                          the current setting is 3 words. Is is defined as num_input paramater in words_prediction_lstm.py
    <newtext_length>    : the length of new text, e.g : 50 words

    For example: words_prediction_lstm test input.txt model_file "had a general" 50