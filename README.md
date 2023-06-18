# MusicEM: Music based on Emotion sequence from the Movie dialogues
EE474 ITM Term Project - Team 2

1. ```data```
 Original & processed data are located here.
  
2. ```emotion_recognizer```
 The emotion recognizer model extracts the emotion sequence from movie dialogues.
 One emotion, (valence, arousal), is extracted for one sentence.

3. ```music_generator```
 The music generator model generates music based on the extracted emotion sequences.
 The ```.csv``` file in the ```emotion_recognizer/output/``` directory is used as the input for this model.
