
#se avete installato python ora o non avete i pacchetti aprite un terminale e digitate
# pip install matplotlib -> attendi installazione
# pip install numpy      -> dovrebbe installarlo con matplotlib ma per riscurezza provate
# pip install pillow     -> installa PIL
#sys lo avete già nel core dell'interprete

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys

IMAGE_WIDTH  = 28
IMAGE_HEIGHT = 28 
MNIST_MAX    = 255
GRAY_SCALE_V = 0
TRANSPARENCY = 1
PATH         = sys.argv[1]

image     = Image.open(PATH)                                 #Passo il path dell'immagine direttamente da js
new_image = Image.new("RGBA", image.size, "WHITE")           #Create a white rgba background
new_image.paste(image, (0, 0), image)                        #Paste the image on the background. Go to the links given below for details.
new_image = new_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))    #Ridimensiono per la rete neurale
new_image.convert('RGBA')
new_image.save(PATH)

data = np.asarray(Image.open(PATH).convert('LA'))    #continuare qui per convertire i dati in vettore

val = 0
net_input_array = []

for i in range(0,len(data)):
    for j in range(0,IMAGE_WIDTH):
        net_input_array.append(MNIST_MAX - data[i][j][GRAY_SCALE_V])



print("\nLa tua immagine vettorizzata pronta per la rete\n")
print(str(net_input_array).replace("[","{").replace("]","}"))


plt.imshow(np.asarray(net_input_array).reshape((28,28)), interpolation="nearest")
plt.show()
print("Len --> ", len(net_input_array))