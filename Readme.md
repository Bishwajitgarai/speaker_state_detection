There is one folder contain images
Target is make them as your require format to mark as speak
I here made a code that modify thr images and convert and save into output folder


pip install pipenv

pipenv shell

pipenv install

#now ready for use

python imageConvert.py  #this will convert your output

now u will ger two folder in output images

now run 
python modelGenerate.py

after a few trainning it will generate 
speaking_detection_model.keras in same folder


now there is a option to make model lighter 

use heavy_to_light_model.py

now you will get speaking_detection_model.tflite in folder that is the lightweight model

